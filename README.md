# Legal RAG Pipeline

A Retrieval-Augmented Generation (RAG) system designed to analyze and query Civil and Criminal legal case documents. This system leverages advanced Natural Language Processing (NLP) techniques to provide accurate context-aware answers from a large corpus of PDF documents.

## Project Overview

This application processes legal documents (PDFs), generates vector embeddings for semantic search, and retrieves relevant context to answer user queries. It supports multiple Large Language Model (LLM) backends, allowing for flexibility in answer generation.

## Key Features

- **PDF Ingestion**: Recursively scans and extracts text from PDFs located in `PDF/CIVIL` and `PDF/CRIMINAL` directories.
- **Hybrid Search Architecture**: Utilizes `multi-qa-mpnet-base-cos-v1` for high-quality semantic embeddings.
- **Cross-Platform Optimization**:
  - **Linux/Mac**: Automatically uses parallel processing (multiprocessing) for rapid data ingestion.
  - **Windows**: Automatically defaults to serial processing with robust crash recovery.
- **Incremental Caching**: Implements a smart caching system (`emb_cache_v2/`) that hashes files based on content and modification time. This ensures that only new or modified files are processed in subsequent runs.
- **Crash Recovery**: Saves progress after every single file during the embedding phase, preventing data loss during long processing tasks.
- **Multi-Backend LLM Support**:
  - Local Models: LLaMA 3, Mistral, Phi-3 (via Ollama).
  - Cloud APIs: GPT-4, GPT-3.5, Claude 3, **Gemini 1.5 Pro**.
- **Automated Metrics**: Calculates ROUGE-L, BLEU, METEOR, BERTScore, and Cosine Similarity for every generated answer to quantify performance.

## Prerequisites

- **Python 3.10+**
- **Ollama**: Required for running local LLMs. Ensure it is installed and running.
- **Google Cloud Credentials (Optional)**: Required only if using the Google Drive Sync feature.

## Installation

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd MAJOR_PROJECT
   ```

2. **Set Up Virtual Environment**

   ```bash
   python3 -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Local Models (Ollama)**
   If using local models, pull them before running the script:

   ```bash
   ollama pull llama3:8b
   ollama pull mistral:7b
   ```

5. **Configure Environment Variables**
   Rename `.env` and populate it with your API keys if using cloud providers:
   ```properties
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ```

## Usage

Run the main application:

```bash
python main.py
```

### Execution Flow

1.  **Data Selection**: Choose to process `civil`, `criminal`, or `both` directories.
2.  **Processing**: The system checks for new files.
    - On first run, it will extract and embed all PDFs.
    - On subsequent runs, it loads vectors effectively from the cache.
3.  **LLM Selection**: Select your desired model (e.g., LLaMA 3, GPT-4) from the menu.
4.  **Querying**: The system runs predefined test queries (or you can modify the script to accept custom input) and displays the answer along with evaluation metrics.

## Utilities

### Google Drive Sync

To download PDFs directly from a Google Drive folder:

1.  Place your `credentials.json` (from Google Cloud Console) in the project root.
2.  Edit `drive_sync.py` to add your generic Folder ID.
3.  Run the script:
    ```bash
    python drive_sync.py
    ```

## Troubleshooting

- **Ollama Not Found**: The script attempts to auto-detect Ollama. If it fails, ensure `ollama` is in your system PATH.
- **Parallel Processing Errors**: On Windows, the system defaults to serial mode to avoid `multiprocessing` spawn errors. Do not force parallel mode on Windows unless configured correctly.
- **Missing Dependencies**: If `psutil` or other modules are missing, reinstall `requirements.txt`.

## License

This project is licensed for educational and research use.
