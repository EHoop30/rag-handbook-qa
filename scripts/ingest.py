"""CLI to ingest the corpus into the configured vector store.

Usage:
    python -m scripts.ingest [corpus_dir]

Reads provider and database settings from the environment (.env). Idempotent:
re-running on an unchanged corpus does not create duplicate chunks.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.config import get_settings
from app.embeddings import build_embedder
from app.ingest import ingest_path
from app.vectorstore import PgVectorStore


def main() -> None:
    corpus = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    settings = get_settings()
    embedder = build_embedder(settings)
    store = PgVectorStore(settings.database_url, dim=embedder.dim)

    print(f"Ingesting {Path(corpus).resolve()} "
          f"(embeddings={settings.embedding_provider}) ...")
    stats = ingest_path(corpus, embedder, store, settings.chunk_size, settings.chunk_overlap)
    print(f"  documents: {stats['documents']}")
    print(f"  chunks upserted: {stats['chunks']}")
    print(f"  total chunks indexed: {stats['total_indexed']}")


if __name__ == "__main__":
    main()
