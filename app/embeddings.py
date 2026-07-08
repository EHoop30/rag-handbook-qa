"""Embedding providers behind one interface.

Swapping the provider is a config change, never a code change. The real
provider (OpenAI) is what you run in production; the fake provider lets the
whole pipeline run offline in tests and in no-key demos, and still produces
*meaningful* retrieval so tests actually exercise ranking rather than mocking
it away.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def embed_query(embedder: Embedder, text: str) -> list[float]:
    return embedder.embed([text])[0]


# ---- Fake (offline, deterministic) -----------------------------------------


class FakeEmbedder:
    """Hashing bag-of-words embedder. No model, no network, fully deterministic.

    Each token is hashed into one of `dim` buckets; the vector is L2-normalized.
    Two texts that share vocabulary land close in cosine space, so retrieval
    behaves sensibly for tests and offline demos without pretending to be a
    real semantic model.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _WORD.findall(text.lower()):
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


# ---- OpenAI ----------------------------------------------------------------


class OpenAIEmbedder:
    """text-embedding-3-small by default (1536 dims). Batches in one request."""

    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = self._DIMS.get(model, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        # Preserve input order (the API guarantees it, but be explicit).
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]


def build_embedder(settings) -> Embedder:
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is unset")
        return OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model)
    if settings.embedding_provider == "fake":
        return FakeEmbedder()
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
