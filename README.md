# Legal RAG Pipeline ⚖️

A Retrieval-Augmented Generation (RAG) system for analyzing Civil and Criminal legal case documents (PDFs). This system uses **MPNet** for embeddings and supports switching between various LLM backends (OpenAI, Anthropic, Gemini, LLaMA 3, Mistral) for answer generation. Metrics (ROUGE, BERTScore, BLEU) are automatically calculated for every run.

## 🚀 Features

- **PDF Ingestion**: Recursively scans and parses PDFs from `PDF/CIVIL` and `PDF/CRIMINAL`.
- **Hybrid Search**: Uses high-performance vector embeddings (`multi-qa-mpnet-base-cos-v1`).
- **Flexible LLM Backend**: Switch instantly between:
  - **Local Models**: LLaMA-3, Mistral, Phi-3 (via Ollama).
  - **Cloud APIs**: GPT-4, GPT-3.5, Claude 3, Gemini 1.5 Pro.
- **Advanced Metrics**: Automatically evaluates answers using ROUGE-L, METEOR, BLEU, BERTScore, and Cosine Similarity.
- **Caching**: Embeddings are cached locally (`emb_cache/`) to speed up subsequent runs.

## 🛠️ Prerequisites

- **Python 3.10+**
- **Ollama** (for local models) installed and available in PATH.

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd MAJOR_PROJECT
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   _Note: First run might take time to download `torch` and transformer models._

4. **Pull Local Models (Ollama)**
   If you plan to use local LLMs, make sure your Ollama app is running and pull the models:

   ```bash
   ollama pull llama3:8b
   ollama pull mistral:7b
   ollama pull phi3:mini
   ```

5. **Set Up API Keys (Optional)**
   If you want to use Cloud LLMs (OpenAI/Claude/Gemini) or Pinecone, rename `.env` and add your keys:
   ```bash
   # .env file
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=AIza...
   PINECONE_API_KEY=...
   ```

## 🏃 Usage

Run the main script:

```bash
python main.py
```

### Steps:

1.  **Select Data**: The script will ask whether to process `civil`, `criminal`, or `both` types of PDFs.
2.  **Date Filtering**: You can choose to process "ALL" years or specific ones (e.g., "2000, 2005-2008").
3.  **LLM Selection**: Choose your desired backend from the menu:
    - `1` GPT-4
    - `4` LLaMA 3 (Local)
    - `7` Gemini
    - ...
4.  **Results**: The system will retrieve context, generate an answer, and display a table of accuracy metrics.
5.  **Logs**: A CSV file (e.g., `metrics_mpnet.csv`) containing detailed logs of the session is saved automatically.

## 📂 Project Structure

```
.
├── PDF/                   # Place your PDF documents here (CIVIL/CRIMINAL folders)
├── emb_cache/             # Stores generated embeddings (auto-created)
├── main.py                # Main application logic
├── requirements.txt       # Python dependencies
├── .env                   # API keys configuration
└── README.md              # Project documentation
```

## ❓ Troubleshooting

- **`ollama: command not found`**: Ensure Ollama is installed. On Windows, the default path is often `C:\Users\<user>\AppData\Local\Programs\Ollama\ollama.exe`. Add this to your System PATH or use the full path.
- **`torch` errors**: If you have GPU availability issues, ensure you installed the correct PyTorch version for your CUDA driver.

## 📜 License

[Your License Here]
