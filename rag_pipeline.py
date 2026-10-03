"""
Universal RAG Pipeline
- Supports: PDF, DOCX, TXT, CSV, XLSX, JSON, HTML, MD
- Auto-detects document structure
- Uses Ollama for embeddings and LLM (no API keys needed)
- Qdrant as vector store (cloud or local)
"""

import os
import re
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
import os

load_dotenv()
# ─────────────────────────────────────────────
# QDRANT CONFIG — set via environment variables
# ─────────────────────────────────────────────
QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
EMBED_DIM      = 768  # nomic-embed-text output dimension


# ─────────────────────────────────────────────
# 1. DOCUMENT LOADERS
# ─────────────────────────────────────────────

def load_txt(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"content": text, "metadata": {"source": path, "type": "txt"}}]


def load_pdf(path: str) -> List[Dict]:
    try:
        import pdfplumber
        docs = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables()
                for table in tables:
                    rows = [" | ".join(str(c) for c in row if c) for row in table if any(row)]
                    text += "\n" + "\n".join(rows)
                if text.strip():
                    docs.append({
                        "content": text.strip(),
                        "metadata": {"source": path, "type": "pdf", "page": i + 1}
                    })
        return docs
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return []


def load_docx(path: str) -> List[Dict]:
    try:
        from docx import Document
        doc = Document(path)
        chunks = []
        current = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                if current:
                    chunks.append("\n".join(current))
                    current = []
            else:
                current.append(text)
        if current:
            chunks.append("\n".join(current))
        return [
            {"content": c, "metadata": {"source": path, "type": "docx", "chunk": i}}
            for i, c in enumerate(chunks) if c.strip()
        ]
    except ImportError:
        logger.error("python-docx not installed. Run: pip install python-docx")
        return []


def load_csv(path: str) -> List[Dict]:
    import csv
    docs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            content = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
            if content.strip():
                docs.append({
                    "content": content,
                    "metadata": {"source": path, "type": "csv", "row": i + 1}
                })
    return docs


def load_xlsx(path: str) -> List[Dict]:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        docs = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue
            headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
            for i, row in enumerate(rows[1:], 1):
                content = " | ".join(
                    f"{headers[j]}: {v}" for j, v in enumerate(row) if v is not None
                )
                if content.strip():
                    docs.append({
                        "content": content,
                        "metadata": {"source": path, "type": "xlsx", "sheet": sheet_name, "row": i}
                    })
        return docs
    except ImportError:
        logger.error("openpyxl not installed. Run: pip install openpyxl")
        return []


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def flatten(obj, prefix=""):
        if isinstance(obj, dict):
            return " | ".join(f"{prefix}{k}: {flatten(v, '')}" for k, v in obj.items())
        elif isinstance(obj, list):
            return "; ".join(flatten(item) for item in obj if item)
        else:
            return str(obj)

    docs = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            content = flatten(item)
            docs.append({"content": content, "metadata": {"source": path, "type": "json", "index": i}})
    else:
        content = flatten(data)
        docs.append({"content": content, "metadata": {"source": path, "type": "json"}})
    return docs


def load_html(path: str) -> List[Dict]:
    try:
        from bs4 import BeautifulSoup
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return [{"content": text, "metadata": {"source": path, "type": "html"}}]
    except ImportError:
        logger.error("beautifulsoup4 not installed. Run: pip install beautifulsoup4")
        return []


def load_markdown(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"content": text, "metadata": {"source": path, "type": "markdown"}}]


LOADERS = {
    ".txt":      load_txt,
    ".pdf":      load_pdf,
    ".docx":     load_docx,
    ".doc":      load_docx,
    ".csv":      load_csv,
    ".xlsx":     load_xlsx,
    ".xls":      load_xlsx,
    ".json":     load_json,
    ".html":     load_html,
    ".htm":      load_html,
    ".md":       load_markdown,
    ".markdown": load_markdown,
}


def load_document(path: str) -> List[Dict]:
    ext = Path(path).suffix.lower()
    loader = LOADERS.get(ext)
    if not loader:
        logger.warning(f"Unsupported file type: {ext} — skipping {path}")
        return []
    logger.info(f"Loading {path} ...")
    return loader(path)


# ─────────────────────────────────────────────
# 2. SMART CHUNKER
# ─────────────────────────────────────────────

def detect_record_pattern(text: str):
    from collections import Counter
    candidates = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*){0,2})\s+is\b', text)
    if not candidates:
        return None
    counts = Counter(candidates)
    most_common_phrase, freq = counts.most_common(1)[0]
    if freq >= 3:
        return most_common_phrase
    return None


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    for sep in separators:
        if sep and sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    current = part
            if current:
                chunks.append(current.strip())
            if overlap > 0 and len(chunks) > 1:
                overlapped = [chunks[0]]
                for i in range(1, len(chunks)):
                    tail = chunks[i - 1][-overlap:]
                    overlapped.append(tail + " " + chunks[i])
                return [c for c in overlapped if c.strip()]
            return [c for c in chunks if c.strip()]
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]


def smart_chunk(doc: Dict, chunk_size: int = 300, overlap: int = 50) -> List[Dict]:
    content = doc["content"]
    meta = doc["metadata"]
    file_type = meta.get("type", "txt")

    if file_type in ("csv", "xlsx"):
        return [doc]

    if file_type == "json" and "index" in meta:
        return [doc]

    if file_type == "txt":
        pattern = detect_record_pattern(content)
        if pattern:
            logger.info(f"Detected record pattern: '{pattern}' — splitting on delimiter")
            raw_chunks = re.split(rf'(?=\b{re.escape(pattern)}\b)', content)
            chunks = [c.strip() for c in raw_chunks if c.strip()]
            if len(chunks) > 1:
                return [
                    {"content": c, "metadata": {**meta, "chunk": i}}
                    for i, c in enumerate(chunks)
                ]

    text_chunks = chunk_text(content, chunk_size, overlap)
    merged = []
    i = 0
    while i < len(text_chunks):
        current = text_chunks[i]
        if len(current) < 100 and i + 1 < len(text_chunks):
            current = current + "\n" + text_chunks[i + 1]
            i += 2
        else:
            i += 1
        merged.append(current)

    return [
        {"content": c, "metadata": {**meta, "chunk": j}}
        for j, c in enumerate(merged)
    ]


# ─────────────────────────────────────────────
# 3. OLLAMA EMBEDDING
# ─────────────────────────────────────────────

def get_ollama_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    import requests
    text = text[:4000] if len(text) > 4000 else text
    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ─────────────────────────────────────────────
# 4. VECTOR STORE (Qdrant)
# ─────────────────────────────────────────────

def get_qdrant_client():
    from qdrant_client import QdrantClient
    # If host contains http:// or https://, use url param instead
    if QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
        url = f"{QDRANT_HOST}:{QDRANT_PORT}"
        client = QdrantClient(
            url=url,
            api_key=QDRANT_API_KEY,
            timeout=60,
        )
    else:
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            timeout=60,
        )
    return client


def get_collection_name(file_path: str) -> str:
    name = Path(file_path).stem
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:40]
    return f"doc_{name}".lower()


def ensure_collection(client, col_name: str):
    """Create Qdrant collection if it doesn't exist."""
    from qdrant_client.models import Distance, VectorParams
    existing = [c.name for c in client.get_collections().collections]
    if col_name not in existing:
        client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {col_name}")


def ingest_documents(
    file_paths: List[str],
    db_path: str = "./rag_db",       # kept for API compatibility, unused with Qdrant
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "nomic-embed-text"
):
    from qdrant_client.models import PointStruct

    client = get_qdrant_client()

    for path in file_paths:
        raw_docs = load_document(path)
        if not raw_docs:
            continue

        all_chunks = []
        for doc in raw_docs:
            chunks = smart_chunk(doc, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)

        if not all_chunks:
            continue

        col_name = get_collection_name(path)
        ensure_collection(client, col_name)

        logger.info(f"Indexing {len(all_chunks)} chunks for {Path(path).name} → '{col_name}'")

        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            points = []
            for chunk in batch:
                embedding = get_ollama_embedding(chunk["content"], embedding_model)
                # Qdrant payload must have only string/int/float/bool values
                payload = {
                    "text":        chunk["content"],
                    "source":      str(chunk["metadata"].get("source", path)),
                    "source_file": Path(path).name,
                    "doc_type":    str(chunk["metadata"].get("type", "")),
                    "chunk_index": int(chunk["metadata"].get("chunk", 0)),
                }
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload,
                ))
            client.upsert(collection_name=col_name, points=points)
            logger.info(f"  Upserted batch {i}–{i + len(batch)}")

        logger.info(f"Done indexing: {Path(path).name}")

    logger.info("All documents ingested.")
    return client


# ─────────────────────────────────────────────
# 5. RETRIEVAL
# ─────────────────────────────────────────────

def retrieve(query: str, client, top_k: int = 8) -> List[Dict]:
    from qdrant_client.models import Filter, FieldCondition, MatchText

    query_embedding = get_ollama_embedding(query)
    query_upper = query.upper()

    # ── List all doc_ collections ──
    all_col_names = [
        c.name for c in client.get_collections().collections
        if c.name.startswith("doc_")
    ]
    if not all_col_names:
        logger.warning("No collections found in Qdrant.")
        return []

    logger.info(f"Query: '{query}'")
    logger.info(f"Collections found: {all_col_names}")

    # ── Step 1: Score each collection with a quick top-2 search ──
    collection_scores = []
    for col_name in all_col_names:
        try:
            results = client.query_points(
                collection_name=col_name,
                query=query_embedding,
                limit=2,
                with_payload=False,
            ).points
            if results:
                best_score = max(r.score for r in results)  # Qdrant: higher = better
                collection_scores.append((col_name, best_score))
        except Exception as e:
            logger.warning(f"Scoring failed for {col_name}: {e}")

    if not collection_scores:
        return []

    # Sort descending — highest score = most relevant
    collection_scores.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Collection scores: {[(n, round(s, 3)) for n, s in collection_scores]}")

    # ── Step 2: Name extraction for person-specific queries ──
    name_results = []
    name_match = re.search(
        r'(?:OF|ABOUT|FOR)\s+([A-Z]+(?:\s[A-Z]+){0,2})(?:\?|$|\s+IS|\s+HAS|\s+WAS)',
        query_upper
    )
    if name_match:
        matched = name_match.group(1).strip()
        non_name_words = {
            "POLICE", "THE", "ORDER", "GOVERNMENT", "DISTRICT", "STATE",
            "DEPARTMENT", "INDIA", "BENGAL", "WEST", "RULES", "ACT",
            "SECTION", "ARTICLE", "CLAUSE", "LIST", "AWARD", "STATION"
        }
        if len(matched.split()) > 3 or len(matched) > 40 or set(matched.split()).issubset(non_name_words):
            name_match = None
            logger.info(f"Rejected name match: {matched}")

    if name_match:
        full_name = name_match.group(1).strip()
        logger.info(f"Extracted name: {full_name}")
        name_words = [w for w in full_name.split() if len(w) > 2]

        # Search in top collection only for name
        top_col = collection_scores[0][0]
        for search_term in [full_name] + name_words:
            try:
                results = client.scroll(
                    collection_name=top_col,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="text", match=MatchText(text=search_term))]
                    ),
                    limit=10,
                    with_payload=True,
                    with_vectors=False,
                )[0]
                for r in results:
                    name_results.append({
                        "content":  r.payload["text"],
                        "metadata": {"source": r.payload.get("source", "")},
                        "distance": 0.0,  # highest priority
                    })
                if name_results:
                    break  # full name matched — no need for word-by-word
            except Exception as e:
                logger.warning(f"Name search failed for '{search_term}': {e}")

    # ── Step 3: Vector search in top 3 collections ──
    all_results = list(name_results)
    seen = {r["content"][:80] for r in all_results}

    for col_name, _ in collection_scores[:3]:
        try:
            results = client.query_points(
                collection_name=col_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
            ).points
            for r in results:
                text = r.payload["text"]
                if text[:80] not in seen:
                    seen.add(text[:80])
                    all_results.append({
                        "content":  text,
                        "metadata": {"source": r.payload.get("source", "")},
                        "distance": 1 - r.score,  # convert similarity to distance
                    })
        except Exception as e:
            logger.warning(f"Vector search failed for {col_name}: {e}")

    # ── Step 4: Keyword search in top 3 collections ──
    stopwords = {
        "what", "is", "the", "of", "a", "an", "in", "at", "for", "to",
        "and", "or", "tell", "me", "give", "find", "show", "who", "where",
        "when", "how", "which", "his", "her", "about", "please", "can"
    }
    query_words = [
        w.lower().strip("?.,;:")
        for w in query.split()
        if w.lower() not in stopwords and len(w) > 3
    ]

    for col_name, _ in collection_scores[:3]:
        if len(all_results) >= top_k * 3:
            break
        for word in query_words[:4]:
            if len(all_results) >= top_k * 3:
                break
            try:
                results = client.scroll(
                    collection_name=col_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="text", match=MatchText(text=word))]
                    ),
                    limit=10,
                    with_payload=True,
                    with_vectors=False,
                )[0]
                for r in results:
                    text = r.payload["text"]
                    if text[:80] not in seen:
                        seen.add(text[:80])
                        all_results.append({
                            "content":  text,
                            "metadata": {"source": r.payload.get("source", "")},
                            "distance": 0.2,
                        })
            except Exception as e:
                logger.warning(f"Keyword search failed for '{word}': {e}")

    all_results.sort(key=lambda x: x["distance"])
    logger.info(f"Final results count: {len(all_results)}, sending top {top_k}")
    for i, d in enumerate(all_results[:top_k]):
        logger.info(f"  Chunk {i}: {d['content'][:80]}")
    return all_results[:top_k]


# ─────────────────────────────────────────────
# 6. OLLAMA LLM ANSWER GENERATION
# ─────────────────────────────────────────────

def generate_answer(query: str, context_docs: List[Dict], model: str = "llama3.2") -> str:
    import requests

    if not context_docs:
        return "I could not find relevant information in the documents."

    context = "\n\n---\n\n".join(
        f"[Source: {d['metadata'].get('source', 'unknown')}]\n{d['content']}"
        for d in context_docs
    )

    prompt = f"""You are a precise information extraction assistant.
The user has asked a specific question. The answer IS present in the context below.
Extract and state it directly and concisely.
Do NOT say the information is missing. Do NOT summarize all records.
Just answer the specific question asked.

Context:
{context}

Question: {query}

Answer (be specific and direct):"""

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE CLASS
# ─────────────────────────────────────────────

class UniversalRAG:
    def __init__(
        self,
        db_path: str = "./rag_db",          # unused with Qdrant, kept for compatibility
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        top_k: int = 8,
    ):
        self.db_path         = db_path
        self.embedding_model = embedding_model
        self.llm_model       = llm_model
        self.chunk_size      = chunk_size
        self.chunk_overlap   = chunk_overlap
        self.top_k           = top_k
        self.client          = None

    def _get_client(self):
        if not self.client:
            self.client = get_qdrant_client()
        return self.client

    def ingest(self, paths: List[str]):
        file_paths = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                for f in path.rglob("*"):
                    if f.suffix.lower() in LOADERS:
                        file_paths.append(str(f))
            elif path.is_file():
                file_paths.append(str(path))
            else:
                logger.warning(f"Path not found: {p}")

        self.client = ingest_documents(
            file_paths,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_model=self.embedding_model,
        )

    def load(self):
        self.client = get_qdrant_client()

    def query(self, question: str) -> Dict[str, Any]:
        client = self._get_client()
        context_docs = retrieve(question, client, self.top_k)
        answer = generate_answer(question, context_docs, self.llm_model)
        return {
            "question": question,
            "answer":   answer,
            "sources":  [
                {"source": d["metadata"].get("source", ""), "score": round(1 - d["distance"], 3)}
                for d in context_docs
            ]
        }

    def interactive(self):
        print("\n=== Universal RAG — Interactive Mode ===")
        print("Type 'exit' to quit.\n")
        self.load()
        while True:
            question = input("You: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not question:
                continue
            result = self.query(question)
            print(f"\nAnswer: {result['answer']}")
            print("Sources:", ", ".join(s["source"] for s in result["sources"]))
            print()