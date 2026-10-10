"""
Pinecone backend — cloud vector store with the same interface as the local
Chroma store (upsert_documents / delete_documents / search / count).

Used when VECTOR_BACKEND=pinecone: on ephemeral hosting (App Platform) the
local Chroma index doesn't survive restarts, and the Sync button needs to
update an index the running agent reads from — Pinecone does both.

Namespaces: one per content-hash batch isn't needed; everything lives in the
default namespace, ids are "<content_hash>:<chunk_index>" (same as Chroma) so
incremental reindex semantics are identical.
"""
from __future__ import annotations

import os
import threading

from app.core.config import PINECONE_API_KEY, PINECONE_INDEX
from app.core.logging import logger
from app.rag.vectorstore import EMBED_MODEL, chunk_text, embed

_index = None
_lock = threading.Lock()
_dimension: int | None = None


def _get_index():
    global _index, _dimension
    if _index is None:
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set — cannot use pinecone backend")
        from pinecone import Pinecone, ServerlessSpec

        # Corporate proxy does SSL inspection with a self-signed chain —
        # same treatment as the Portkey LLM client in core/config.py.
        # Verification stays ON unless PINECONE_INSECURE_TLS=1 is set
        # explicitly (local dev behind the proxy only — never on hosting).
        insecure = os.getenv("PINECONE_INSECURE_TLS", "0") == "1"
        pc = Pinecone(api_key=PINECONE_API_KEY, ssl_verify=not insecure)
        if not pc.has_index(PINECONE_INDEX):
            probe = len(embed(["dim probe"])[0])
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=probe,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _index = pc.Index(PINECONE_INDEX)
        _dimension = _index.describe_index_stats().get("dimension")
    return _index


def upsert_documents(records: list[dict]) -> dict:
    """Same contract as the Chroma store: chunk → embed → upsert."""
    if not records:
        return {"chunks_added": 0}
    index = _get_index()
    vectors: list[dict] = []
    for rec in records:
        chunks = chunk_text(rec["content"])
        if not chunks:
            continue
        embeddings = embed(chunks)
        branches = ",".join(rec["branches"])[:500]
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{rec['content_hash']}:{i}",
                "values": vec,
                "metadata": {
                    "text": chunk[:4000],  # Pinecone metadata size cap
                    "content_hash": rec["content_hash"],
                    "path": rec["path"],
                    "branches": branches,
                    "chunk_index": i,
                    "source": "kb",
                },
            })
    with _lock:
        block = 96  # Pinecone recommends ≤100 vectors per upsert
        for i in range(0, len(vectors), block):
            index.upsert(vectors=vectors[i : i + block])
    logger.info("pinecone: upserted %d vectors", len(vectors))
    return {"chunks_added": len(vectors)}


def delete_documents(content_hashes: list[str]) -> int:
    """Delete all chunk vectors of the given unique documents."""
    if not content_hashes:
        return 0
    index = _get_index()
    with _lock:
        for h in content_hashes:
            index.delete(prefix=f"{h}:", delete_all=False)
    logger.info("pinecone: deleted chunks for %d documents", len(content_hashes))
    return len(content_hashes)


def delete_all() -> None:
    """Wipe the index (full rebuild path)."""
    _get_index().delete(delete_all=True)
    logger.info("pinecone: index wiped")


def search(query: str, k: int = 6) -> list[dict]:
    """Cosine top-k, same return shape as the Chroma store."""
    index = _get_index()
    stats = index.describe_index_stats()
    if stats.get("total_vector_count", 0) == 0:
        return []
    res = index.query(
        vector=embed([query])[0],
        top_k=k,
        include_metadata=True,
    )
    out: list[dict] = []
    for m in res.get("matches", []):
        meta = m.get("metadata", {})
        out.append({
            "text": meta.get("text", ""),
            "path": meta.get("path", ""),
            "branches": meta.get("branches", ""),
            "source": meta.get("source", "kb"),
            "score": round(m.get("score", 0.0), 4),
        })
    return out


def count() -> int:
    try:
        return _get_index().describe_index_stats().get("total_vector_count", 0)
    except Exception as exc:
        logger.warning("pinecone: count failed (%s)", exc)
        return 0
