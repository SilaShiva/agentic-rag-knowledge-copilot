"""
Vector store — Chroma (persistent, local) + fastembed MiniLM-L6-v2 (ONNX, free).

Chunk-level records: each unique kb document is split into overlapping
word chunks; metadata carries content_hash + path + branches so incremental
reindex can delete/re-embed only what changed.
"""
from __future__ import annotations

import threading

import chromadb
from chromadb.config import Settings

from app.core.config import CHROMA_DIR, RETRIEVAL_K
from app.core.logging import logger

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "kb_knowledge"

_client: chromadb.ClientAPI | None = None
_embedder = None
_lock = threading.Lock()


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )
    return _client


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    return [[float(x) for x in v] for v in _get_embedder().embed(texts)]


def upsert_documents(records: list[dict]) -> dict:
    """
    records: [{content_hash, path, branches, content}] — chunk, embed, upsert.

    Deletion of removed docs is handled by delete_documents(). Returns counts.
    """
    if not records:
        return {"chunks_added": 0}
    collection = get_collection()
    with _lock:
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embeddings: list[list[float]] = []
        for rec in records:
            chunks = chunk_text(rec["content"])
            for i, chunk in enumerate(chunks):
                ids.append(f"{rec['content_hash']}:{i}")
                documents.append(chunk)
                metadatas.append({
                    "content_hash": rec["content_hash"],
                    "path": rec["path"],
                    "branches": ",".join(rec["branches"])[:500],
                    "chunk_index": i,
                    "source": "kb",
                })
        embeddings = embed(documents)
        # Chroma caps batch sizes; go in blocks
        block = 128
        for i in range(0, len(ids), block):
            collection.upsert(
                ids=ids[i : i + block],
                documents=documents[i : i + block],
                metadatas=metadatas[i : i + block],
                embeddings=embeddings[i : i + block],
            )
    logger.info("vectorstore: upserted %d chunks", len(ids))
    return {"chunks_added": len(ids)}


def delete_documents(content_hashes: list[str]) -> int:
    """Remove all chunks of the given unique documents. Returns n deleted."""
    if not content_hashes:
        return 0
    collection = get_collection()
    collection.delete(where={"content_hash": {"$in": content_hashes}})
    logger.info("vectorstore: deleted chunks for %d documents", len(content_hashes))
    return len(content_hashes)


def search(query: str, k: int = RETRIEVAL_K) -> list[dict]:
    """Cosine top-k. Returns [{text, path, branches, score, source}]."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    q_vec = embed([query])[0]
    res = collection.query(
        query_embeddings=[q_vec],
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({
            "text": doc,
            "path": meta.get("path", ""),
            "branches": meta.get("branches", ""),
            "source": meta.get("source", "kb"),
            "score": round(1.0 - dist, 4),  # cosine distance → similarity
        })
    return out


def count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0
