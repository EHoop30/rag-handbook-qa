"""Shared fixtures. Everything here is offline: fake providers, in-memory store.

The `service` fixture ingests the real corpus so tests exercise the true
ingestion and retrieval path, just without a database or network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.embeddings import FakeEmbedder
from app.ingest import ingest_path
from app.llm import FakeAnswerer
from app.retrieval import RagService
from app.vectorstore import InMemoryVectorStore

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def service(embedder, store) -> RagService:
    ingest_path(CORPUS, embedder, store)
    return RagService(embedder, store, FakeAnswerer())
