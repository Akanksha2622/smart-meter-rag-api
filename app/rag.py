"""Core RAG logic: load docs -> chunk -> embed -> store -> retrieve -> answer."""
import os
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader

DOCS_DIR = Path(os.getenv("DOCS_DIR", "data/docs"))
DB_DIR = os.getenv("DB_DIR", "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
# Model names change over time: check https://console.groq.com/docs/models
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
COLLECTION = "docs"

_embedder = None  # loaded lazily so tests/imports stay fast


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks (measured in characters)."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def build_index(chunk_size: int = 800, overlap: int = 150) -> int:
    """Read every .txt/.md/.pdf in DOCS_DIR and (re)build the vector index."""
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    ids, texts, metas = [], [], []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        for i, chunk in enumerate(chunk_text(read_file(path), chunk_size, overlap)):
            ids.append(f"{path.name}-{i}")
            texts.append(chunk)
            metas.append({"source": path.name, "chunk": i})

    if not texts:
        raise RuntimeError(f"No documents found in {DOCS_DIR}")
    embeddings = get_embedder().encode(texts).tolist()
    col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
    return len(texts)


def retrieve(question: str, k: int = 3) -> list[dict]:
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        col = client.get_collection(COLLECTION)
    except Exception as exc:
        raise RuntimeError("Index not built. Run: python -m app.ingest") from exc
    emb = get_embedder().encode([question]).tolist()
    res = col.query(query_embeddings=emb, n_results=k)
    return [
        {"text": doc, "source": meta["source"], "chunk": meta["chunk"]}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]


def answer_question(question: str, k: int = 3) -> dict:
    chunks = retrieve(question, k)
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know. "
        "Mention the source file name(s) you used.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return {"answer": resp.choices[0].message.content, "sources": chunks}
