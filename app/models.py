"""Domain models shared across ingestion, retrieval, and the API.

Chunk is the internal unit that flows through the pipeline. The `*Response`
models are the public API contract, kept deliberately small and stable.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One retrievable passage of a document.

    `id` is content-derived so re-ingesting an unchanged document produces
    the same id, which is what makes ingestion idempotent (upsert, no dupes).
    """

    id: str
    doc_id: str
    title: str
    text: str
    chunk_index: int
    embedding: list[float] | None = None

    @staticmethod
    def make_id(doc_id: str, chunk_index: int, text: str) -> str:
        h = hashlib.sha256(f"{doc_id}:{chunk_index}:{text}".encode()).hexdigest()
        return h[:16]


class SearchResult(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]; higher is closer


# ---- API contract ----------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
