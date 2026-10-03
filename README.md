# Universal RAG Pipeline

No API keys. No OpenAI. Runs 100% locally using Ollama.

## Supported file types

| Format | Extension |
|--------|-----------|
| Plain text | .txt |
| PDF | .pdf |
| Word document | .docx, .doc |
| CSV | .csv |
| Excel | .xlsx, .xls |
| JSON | .json |
| HTML | .html, .htm |
| Markdown | .md, .markdown |

---

## Setup

### 1. Install Ollama
Download from https://ollama.com and run:
```bash
ollama pull nomic-embed-text   # embedding model
ollama pull llama3             # LLM for answering
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Option A — Web UI
```bash
uvicorn app:app --reload
```
Open http://localhost:8000 in your browser.
- Upload any documents using the drag-and-drop zone
- Click "Ingest documents"
- Type your question and click Ask

---

### Option B — CLI

Ingest a file or folder:
```bash
python cli.py ingest documents/employees.txt
python cli.py ingest documents/        # entire folder
```

Ask a single question:
```bash
python cli.py query "What is BASUDEB SANYAL's phone number?"
```

Interactive chat mode:
```bash
python cli.py chat
```

---

### Option C — Python API

```python
from rag_pipeline import UniversalRAG

rag = UniversalRAG(
    embedding_model="nomic-embed-text",
    llm_model="llama3",
    chunk_size=500,
    chunk_overlap=50,
    top_k=3,
)

# Ingest one or more files/folders
rag.ingest(["employees.txt", "reports/", "data.csv"])

# Ask a question
result = rag.query("What is BASUDEB SANYAL's rank?")
print(result["answer"])
print(result["sources"])
```

---

## Configuration

You can set these via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| EMBED_MODEL | nomic-embed-text | Ollama embedding model |
| LLM_MODEL | llama3 | Ollama LLM for answering |
| CHUNK_SIZE | 500 | Characters per chunk |
| CHUNK_OVERLAP | 50 | Overlap between chunks |
| TOP_K | 3 | Number of chunks retrieved per query |

Example:
```bash
LLM_MODEL=mistral TOP_K=5 uvicorn app:app --reload
```

---

## How chunking works

The pipeline auto-detects the best chunking strategy per document type:

- **CSV / XLSX** — each row is its own chunk. No splitting needed.
- **JSON** — each array item is its own chunk.
- **TXT with repeating records** — detects patterns (like repeated field names) and splits on record boundaries automatically.
- **PDF / DOCX / HTML / Markdown** — recursive split: tries paragraph → line → sentence → word boundaries in order, with configurable overlap.

You never need to configure this manually. Drop in any file type and it adapts.

---

## Reset

To clear the database and start fresh:
- Web UI: click "Reset database"
- CLI: delete the `./rag_db` folder manually

---

## Project structure

```
universal_rag/
├── rag_pipeline.py   # core: loaders, chunker, embedder, retriever, LLM
├── app.py            # FastAPI web interface
├── cli.py            # command-line interface
├── requirements.txt  # dependencies
├── static/
│   └── index.html    # web UI
└── README.md
```
