# ===============================================
# RAG PIPELINE v4 (OPTIMIZED: PARALLEL + ROBUST)
# ===============================================
import os
import re
import time
import shutil
import warnings
import psutil
import pickle
import hashlib
import subprocess
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Some weights of the model")

# NLP / metrics libs
import nltk
for res in ["punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# ENVIRONMENT
# =========================================================
from dotenv import load_dotenv
load_dotenv()

# -----------------------
# OLLAMA AUTO-FIX
# -----------------------
# Try to find Ollama if not in PATH
def find_ollama():
    # 1. Check PATH
    if shutil.which("ollama"):
        return "ollama"
    
    # 2. Check known Windows path
    default_path = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Ollama" / "ollama.exe"
    if default_path.exists():
        return str(default_path)
    
    return "ollama" # Fallback hope

OLLAMA_CMD = find_ollama()
print(f"Ollama detected at: {OLLAMA_CMD}")

# -----------------------
# GPU / device
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | torch {torch.__version__}")

# -----------------------
# PDF root
# -----------------------
BASE_DIR = Path("PDF")

# -----------------------
# PDF Parsing (Helper for Parallel)
# -----------------------
try:
    import fitz  # PyMuPDF
    PDF_LIBRARY = "pymupdf"
except Exception:
    from pypdf import PdfReader
    PDF_LIBRARY = "pypdf"

def extract_chunks_from_file(pdf_path_str: str, chunk_size=200):
    """
    Standalone function for multiprocessing. 
    Must take string paths, not Path objects, for maximum pickling compatibility.
    """
    try:
        pdf_path = Path(pdf_path_str)
        if PDF_LIBRARY == "pymupdf":
            doc = fitz.open(str(pdf_path))
            text = "\n".join([page.get_text("text") for page in doc])
            doc.close()
        else:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            text = "\n".join([p.extract_text() or "" for p in reader.pages])

        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.?!])\s+", text)

        chunks = []
        buf = ""
        for sent in sentences:
            if len((buf + " " + sent).split()) <= chunk_size:
                buf = (buf + " " + sent).strip()
            else:
                if buf:
                    chunks.append(buf.strip())
                buf = sent.strip()
        if buf:
            chunks.append(buf.strip())
        return chunks
    except Exception as e:
        # print(f"⚠ Error reading {pdf_path}: {e}")
        return []

# =========================================================
# INCREMENTAL CACHING UTILS
# =========================================================
CACHE_DIR = Path("emb_cache_v2")

def get_file_hash(file_path: Path) -> str:
    stat = file_path.stat()
    sig = f"{file_path.absolute()}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(sig.encode()).hexdigest()

def save_file_cache(cache_path: Path, vectors: np.ndarray, texts: List[str]):
    np.savez_compressed(cache_path, vectors=vectors, texts=texts)

def load_file_cache(cache_path: Path):
    data = np.load(cache_path, allow_pickle=True)
    return data["vectors"], data["texts"]

# =========================================================
# CORE LOGIC: OPTIMIZED PROCESSING
# =========================================================
def process_and_embed_incrementally(pdf_files: List[Path], model_obj):
    CACHE_DIR.mkdir(exist_ok=True)
    
    # 1. Identify which files need processing
    files_to_process = []  # [(pdf, cache_path), ...]
    cached_files = []      # [(pdf, cache_path), ...]
    
    print(f"\nScanning {len(pdf_files)} files for cache...")
    for pdf in pdf_files:
        fhash = get_file_hash(pdf)
        cpath = CACHE_DIR / f"{fhash}.npz"
        if cpath.exists():
            cached_files.append(cpath)
        else:
            files_to_process.append((pdf, cpath))

    # 2. Extract & Embed (Auto-Switch based on OS)
    new_chunks_map = {} # path -> chunks
    
    # Check OS: 'posix' = Linux/Mac (Parallel Safe), 'nt' = Windows (Serial Safer)
    MAX_WORKERS = os.cpu_count() if os.name == 'posix' else 1
    
    if files_to_process:
        if MAX_WORKERS > 1:
            print(f"\n⚡ Parallel Extracting {len(files_to_process)} NEW files using {MAX_WORKERS} cores (Linux/Mac)...")
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_file = {
                    executor.submit(extract_chunks_from_file, str(pdf)): (pdf, cpath)
                    for pdf, cpath in files_to_process
                }
                for future in tqdm(as_completed(future_to_file), total=len(files_to_process), desc="Extracting"):
                    pdf, cpath = future_to_file[future]
                    try:
                        chunks = future.result()
                        if chunks:
                            new_chunks_map[cpath] = chunks
                    except Exception as e:
                        print(f"Parallel Error {pdf}: {e}")
        else:
            print(f"\n🐢 Serial Extracting {len(files_to_process)} NEW files (Windows Safe Mode)...")
            for pdf, cpath in tqdm(files_to_process, desc="Extracting"):
                chunks = extract_chunks_from_file(str(pdf))
                if chunks:
                    new_chunks_map[cpath] = chunks

    # 3. Embedding loop (chunks -> vectors)
    # We process in batches and SAVE FREQUENTLY (Crash Recovery)
    if new_chunks_map:
        print(f"\n🔹 Embedding {len(new_chunks_map)} files (saving every step)...")
        
        keys = list(new_chunks_map.keys())
        
        for cpath in tqdm(keys, desc="Embedding & Saving"):
            chunks = new_chunks_map[cpath]
            if not chunks: continue
                
            # Embed this file's chunks
            vecs = model_obj.encode(chunks, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
            
            # Save IMMEDIATELY
            save_file_cache(cpath, vecs, chunks)
            cached_files.append(cpath) # Mark as done

    # 4. Load everything from cache

    # 4. Load everything from cache
    print("\n📦 Loading all data from cache...")
    all_vectors = []
    all_texts = []
    
    for cpath in tqdm(cached_files, desc="Loading Cache"):
        try:
            v, t = load_file_cache(cpath)
            if len(v) > 0:
                all_vectors.append(v)
                all_texts.extend(t)
        except Exception as e:
            print(f"Corrupt cache {cpath.name}, skipping.")

    if not all_vectors:
        return np.array([]), []

    final_matrix = np.vstack(all_vectors)
    return final_matrix, all_texts

def gather_pdfs(base: Path, folder="CIVIL", years=None, all_flag=True):
    folder_path = base / folder
    if not folder_path.exists(): return []
    pdfs = []
    if all_flag or years is None:
        pdfs = list(folder_path.rglob("*.pdf"))
    else:
        for y in years:
            p = folder_path / str(y)
            if p.exists(): pdfs.extend(p.rglob("*.pdf"))
    return pdfs

# =========================================================
# LLM & RETRIEVAL
# =========================================================
def generate_answer_llm(prompt: str, llm_backend: str) -> str:
    prompt = prompt[:16000]
    
    if llm_backend == "gemini":
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY: return "[Error: GOOGLE_API_KEY missing]"
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            out = model.generate_content(prompt)
            return out.text.strip()
        except Exception as e: return f"[Gemini Error: {e}]"

    if llm_backend in ["gpt4", "gpt35"]:
        try:
            from openai import OpenAI
            client = OpenAI()
            model = "gpt-4o" if llm_backend == "gpt4" else "gpt-3.5-turbo"
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
            return resp.choices[0].message.content.strip()
        except Exception as e: return f"[OpenAI Error: {e}]"

    if llm_backend == "claude":
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(model="claude-3-haiku-20240307", max_tokens=512, temperature=0, messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text.strip()
        except Exception as e: return f"[Claude Error: {e}]"

    if llm_backend in ["llama3", "mistral", "phi3"]:
        import requests
        model_map = {"llama3": "llama3:8b", "mistral": "mistral:7b", "phi3": "phi3:mini"}
        # Use our auto-detected OLLAMA if needed for CLI, but here it's HTTP API (localhost:11434)
        # Assuming Ollama app is running in background.
        try:
            r = requests.post("http://localhost:11434/api/generate", json={"model": model_map.get(llm_backend, "llama3"), "prompt": prompt, "stream": False}, timeout=120)
            r.raise_for_status()
            return r.json()["response"].strip()
        except Exception as e: return f"[Ollama Error: {e}]"

    return ""

def retrieve_answer(query, model_obj, vectors, texts_list, llm_backend, top_k=5):
    q_vec = model_obj.encode([query])[0]
    sims = cosine_similarity([q_vec], vectors)[0]
    top_ids = np.argsort(sims)[-top_k:][::-1]
    retrieved = [texts_list[i] for i in top_ids]
    context = "\n\n".join(retrieved)
    
    if llm_backend and llm_backend != "none":
        prompt = f"Use ONLY the context below to answer.\n\nContext:\n{context}\n\nQ: {query}\nA:"
        answer = generate_answer_llm(prompt, llm_backend)
    else:
        answer = retrieved[0] if retrieved else ""

    return answer, retrieved, np.array(q_vec, dtype=float)

def evaluate_advanced(preds, gts, retrieved_list, q_vecs, embedder_obj, latencies_r, latencies_g, vectors, texts_list, log_file="metrics_log.csv"):
    df_list = []
    rouge = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    for i, (pred, gt, ret, qv) in enumerate(zip(preds, gts, retrieved_list, q_vecs)):
        r = rouge.score(gt, pred)
        bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=smoothie)
        meteor = meteor_score([gt.split()], pred.split())
        try:
            P, R, F1 = bert_score([pred], [gt], lang="en", rescale_with_baseline=True)
            bert_f1 = float(F1[0])
        except Exception: bert_f1 = 0.0

        if len(ret) > 0:
            ret_vec = embedder_obj.encode([ret[0]])[0] 
            cosine_sim = float(cosine_similarity([qv], [ret_vec])[0][0])
        else: cosine_sim = 0.0

        cpu_use = psutil.cpu_percent()
        ram_use_gb = psutil.virtual_memory().used / (1024**3)

        df_list.append({
            "QueryID": i,
            "Answer": pred[:50] + "...",
            "ROUGE-L": round(r["rougeL"].fmeasure, 3),
            "BLEU": round(bleu, 3),
            "METEOR": round(meteor, 3),
            "BERT-F1": round(bert_f1, 3),
            "CosSim": round(cosine_sim, 3),
            "Latency(s)": round(latencies_r[i] + latencies_g[i], 2),
            "RAM(GB)": round(ram_use_gb, 2)
        })

    df = pd.DataFrame(df_list)
    df.to_csv(log_file, index=False)
    print(f"\nAdvanced metrics saved to {log_file}")
    return df

# =========================================================
# MAIN
# =========================================================
def main():
    print("\n--- RAG PIPELINE v4 (OPTIMIZED) ---")
    
    # Selection
    folder_choice = input("Which PDF type? (civil/criminal/both) [both]: ").strip().lower() or "both"
    process_all = input("Process ALL PDFs? (y/n) [y]: ").strip().lower() or "y"
    
    years_list = None
    if process_all != "y":
        year_input = input("Enter years (e.g., 2000, 2002-2005): ").strip()
        years_list = []
        for token in year_input.split(","):
            if "-" in token:
                a, b = map(int, token.split("-"))
                years_list.extend(range(a, b + 1))
            else:
                years_list.append(int(token))

    pdf_files = []
    if folder_choice in ["civil", "both"]:
        pdf_files += gather_pdfs(BASE_DIR, "CIVIL", years_list, process_all == "y")
    if folder_choice in ["criminal", "both"]:
        pdf_files += gather_pdfs(BASE_DIR, "CRIMINAL", years_list, process_all == "y")

    print(f"\nTotal PDFs found: {len(pdf_files)}")
    if not pdf_files: return

    # Model
    model_name = "multi-qa-mpnet-base-cos-v1"
    print(f"\nLoading Embedding Model: {model_name}...")
    model_obj = SentenceTransformer(model_name)

    # PROCESS
    t0 = time.time()
    vectors, texts = process_and_embed_incrementally(pdf_files, model_obj)
    t1 = time.time()
    print(f"\nCompleted in {t1-t0:.2f}s | Vectors: {vectors.shape}")

    if len(vectors) == 0: return

    # LLM Loop
    while True:
        print("\n--- LLM SELECTION ---")
        print("1) GPT-4 (gpt4)")
        print("2) GPT-3.5 (gpt35)")
        print("3) Claude 3 Haiku (claude)")
        print("4) LLaMA 3 (llama3)")
        print("5) Mistral (mistral)")
        print("6) Phi-3 (phi3)")
        print("7) Gemini (gemini)")
        print("8) None (Retrieval Only)")
        print("q) Quit")
        
        llm_input = input("Choice: ").strip().lower()
        if llm_input in ["q", "quit"]: break
        
        llm_map = {
            "1": "gpt4", "2": "gpt35", "3": "claude", 
            "4": "llama3", "5": "mistral", "6": "phi3", "7": "gemini", "8": "none"
        }
        llm_backend = llm_map.get(llm_input, llm_input) 

        print(f"\nRunning tests with LLM={llm_backend}")
        queries = ["Explain the 2001 SC/ST promotion case.", "Summarize a criminal case discussed in the documents."]
        gts = ["SC/ST promotion case details...", "Criminal case summary..."]
        
        preds, ret, q_vecs, lat_r, lat_g = [], [], [], [], []

        for q in queries:
            t0 = time.time()
            ans, r, qv = retrieve_answer(q, model_obj, vectors, texts, llm_backend)
            t1 = time.time()
            print(f"\nQ: {q}\nA: {ans[:200]}...")
            preds.append(ans)
            ret.append(r)
            q_vecs.append(qv)
            lat_r.append(0.5)
            lat_g.append(t1 - t0 - 0.5)

        evaluate_advanced(preds, gts, ret, q_vecs, model_obj, lat_r, lat_g, vectors, texts)
        
        input("\nPress Enter to continue loop...")

if __name__ == "__main__":
    # Windows Mulitprocessing support
    import multiprocessing
    multiprocessing.freeze_support()
    main()
