# Project Technical Summary

## Architectural Overview

This project implements a state-of-the-art Retrieval-Augmented Generation (RAG) pipeline tailored for the legal domain. It addresses the challenge of querying vast amounts of unstructured legal text (PDFs) by converting them into a structured vector space, allowing for semantic search and context-aware answer generation.

## Component Breakdown

### 1. Data Ingestion Layer (`main.py`)

- **PDF Parsing**: Utilizes `PyMuPDF` (fitz) or `pypdf` to extract raw text from documents.
- **Chunking Strategy**: Implements a sliding window approach with a chunk size of 200 words. This ensures that context is preserved within vector limits while adhering to the input constraints of embedding models.
- **Smart Caching**:
  - Instead of monolithic pickling, the system uses a **Per-File Caching** strategy.
  - Identification: `MD5(File Path + Size + Modification Time)`.
  - Storage: Compressed NumPy arrays (`.npz`) in `emb_cache_v2/`.
  - Benefit: Enables O(1) checks for file updates and allows for immediate crash recovery.

### 2. Vector Embedding Layer

- **Model**: `multi-qa-mpnet-base-cos-v1` from SentenceTransformers.
- **Reasoning**: MPNet provides superior semantic capture compared to faster models like MiniLM, which is critical for legal nuance.
- **Inference**:
  - **Linux**: Uses `ProcessPoolExecutor` to spin up N worker processes (where N = CPU cores), extracting text in parallel.
  - **Windows**: Fallback to serial execution to maintain stability (due to lack of `fork()` system call support).
  - **Batching**: Embeddings are computed in batches of 64 to optimize matrix operations on the CPU/GPU.

### 3. Retrieval Layer

- **Algorithm**: Cosine Similarity.
- **Process**:
  1.  User Query is embedded into a vector $Q$.
  2.  Detailed similarity scores are computed against the matrix of Document Vectors $V$.
  3.  Top-K (default 5) highest scoring chunks are retrieved.
- **Optimization**: All embeddings are loaded into RAM as a unified NumPy matrix for extremely fast vector operations, even with thousands of documents.

### 4. Generation Layer (LLM)

- **Design**: Agnostic Interface. The system creates a standard prompt:
  ```text
  Use ONLY the context below to answer.
  Context: {retrieved_chunks}
  Q: {user_query}
  A:
  ```
- **Backends**:
  - **Ollama**: Interacts via local HTTP API (`localhost:11434`), supporting offline usage with models like LLaMA 3.
  - **OpenAI/Anthropic/Google**: Uses official SDKs for cloud-based inference when higher reasoning capabilities are required (GPT-4, Claude 3, Gemini).

### 5. Evaluation Layer

- **Metrics**:
  - **ROUGE-L**: Measures longest common subsequence (structural similarity).
  - **BLEU**: Measures n-gram overlap (precision).
  - **BERTScore**: Uses contextual embeddings to measure semantic similarity, not just word overlap.
  - **Cosine Similarity**: Measures vector distance between the generated answer and the retrieved context.
- **Logging**: All metrics are logged to `.csv` files for longitudinal performance tracking.

## Future Roadmap

1.  **Vector Database**: Migrating from in-memory NumPy arrays to **FAISS** or **ChromaDB** for scalability beyond 10,000 documents.
2.  **Hybrid Search**: Implementing BM25 (Keyword Search) alongside Vector Search (Semantic) to improved retrieval of exact legal clauses/sections.
3.  **UI Implementation**: Developing a Streamlit or React frontend to replace the CLI.

## Directory Structure

```
root/
├── PDF/                  # Raw Documents
├── emb_cache_v2/         # Vector Store (Local filesystem)
├── main.py              # Orchestrator
├── drive_sync.py        # Utility for Cloud Import
└── metrics_log.csv      # Performance Database
```
