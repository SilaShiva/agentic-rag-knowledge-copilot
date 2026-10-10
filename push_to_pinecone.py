"""One-time migration: local Chroma index → Pinecone.

Reads the content-hash manifest (data/index_state.json), loads each unique
document's chunks from the local Chroma store, and upserts them into Pinecone
under the same id scheme (<content_hash>:<chunk_index>).

Run from the project root with the local index already built:
    python push_to_pinecone.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core import config
from app.core.logging import logger
from app.rag import vectorstore
from app.rag.pinecone_store import _get_index


def main() -> int:
    state = json.loads(config.INDEX_STATE_PATH.read_text(encoding="utf-8"))
    print(f"manifest: {len(state)} unique documents")

    # Pull all chunk rows out of local Chroma in pages. The data predates the
    # kb rename, so it lives in the legacy "cg_sage_knowledge" collection —
    # fall back to it when the current one is empty.
    client = vectorstore.get_client()
    collection = vectorstore.get_collection()
    if collection.count() == 0:
        legacy = client.get_collection("cg_sage_knowledge")
        print(f"current collection empty — using legacy ({legacy.count()} chunks)")
        collection = legacy
    total = collection.count()
    print(f"local chroma chunks: {total}")

    index = _get_index()
    stats = index.describe_index_stats()
    print(f"pinecone index: {config.PINECONE_INDEX} — "
          f"{stats.get('total_vector_count', 0)} vectors, "
          f"dim {stats.get('dimension')}")

    page_size = 200
    vectors = []
    offset = 0
    migrated = 0
    while offset < total:
        got = collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=page_size,
            offset=offset,
        )
        docs = got["documents"]
        metas = got["metadatas"]
        embs = got["embeddings"]
        if not docs:
            break
        for doc_id, doc, meta, emb in zip(got["ids"], docs, metas, embs):
            vectors.append({
                "id": doc_id,  # already "<content_hash>:<chunk_index>"
                "values": [float(x) for x in emb],
                "metadata": {
                    "text": (doc or "")[:4000],
                    "content_hash": meta.get("content_hash", ""),
                    "path": meta.get("path", ""),
                    "branches": meta.get("branches", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "source": "kb",
                },
            })
        offset += len(docs)
        if len(vectors) >= 96:
            index.upsert(vectors=vectors)
            migrated += len(vectors)
            print(f"  upserted {migrated}/{total}")
            vectors = []
    if vectors:
        index.upsert(vectors=vectors)
        migrated += len(vectors)
        print(f"  upserted {migrated}/{total}")

    print(f"DONE: {migrated} vectors in Pinecone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
