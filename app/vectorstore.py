"""Vector storage behind one interface.

`InMemoryVectorStore` keeps everything in a dict and does cosine in numpy;
it needs no database, so unit tests and the eval harness run instantly.
`PgVectorStore` is the production backend: Postgres + pgvector with an HNSW
index and cosine distance. Both satisfy the same `VectorStore` protocol, so
ingestion, retrieval, and the API never know which one they are talking to.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.models import Chunk, SearchResult


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> int: ...
    def search(self, query_embedding: list[float], k: int) -> list[SearchResult]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


# ---- In-memory -------------------------------------------------------------


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> int:
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"chunk {c.id} has no embedding")
            self._chunks[c.id] = c  # keyed by content hash: re-upsert is a no-op
        return len(chunks)

    def search(self, query_embedding: list[float], k: int) -> list[SearchResult]:
        if not self._chunks:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        results: list[SearchResult] = []
        for c in self._chunks.values():
            v = np.asarray(c.embedding, dtype=np.float32)
            denom = (np.linalg.norm(v) or 1.0) * q_norm
            score = float(np.dot(v, q) / denom)
            results.append(SearchResult(chunk=c, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()


# ---- pgvector --------------------------------------------------------------


class PgVectorStore:
    """Postgres + pgvector. Cosine distance via the `<=>` operator.

    The `chunks.embedding` column is fixed to `dim`, so the provider used at
    ingestion time must match the one used at query time (a 1536-dim OpenAI
    vector cannot be compared against a 256-dim fake vector). The schema and
    HNSW index are created on first construction if absent.
    """

    def __init__(self, dsn: str, dim: int) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._psycopg = psycopg
        self._register = register_vector
        self._dsn = dsn
        self.dim = dim
        self._ensure_schema()

    def _conn(self):
        conn = self._psycopg.connect(self._dsn, autocommit=True)
        self._register(conn)
        return conn

    def _ensure_schema(self) -> None:
        conn = self._psycopg.connect(self._dsn, autocommit=True)
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._register(conn)
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          TEXT PRIMARY KEY,
                    doc_id      TEXT NOT NULL,
                    title       TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    chunk_index INT  NOT NULL,
                    embedding   vector({self.dim}) NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        finally:
            conn.close()

    def upsert(self, chunks: list[Chunk]) -> int:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                for c in chunks:
                    if c.embedding is None:
                        raise ValueError(f"chunk {c.id} has no embedding")
                    cur.execute(
                        """
                        INSERT INTO chunks (id, doc_id, title, text, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            doc_id = EXCLUDED.doc_id,
                            title = EXCLUDED.title,
                            text = EXCLUDED.text,
                            chunk_index = EXCLUDED.chunk_index,
                            embedding = EXCLUDED.embedding
                        """,
                        (
                            c.id,
                            c.doc_id,
                            c.title,
                            c.text,
                            c.chunk_index,
                            np.asarray(c.embedding, dtype=np.float32),
                        ),
                    )
            return len(chunks)
        finally:
            conn.close()

    def search(self, query_embedding: list[float], k: int) -> list[SearchResult]:
        conn = self._conn()
        try:
            q = np.asarray(query_embedding, dtype=np.float32)
            rows = conn.execute(
                """
                SELECT id, doc_id, title, text, chunk_index,
                       1 - (embedding <=> %s) AS score
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (q, q, k),
            ).fetchall()
            return [
                SearchResult(
                    chunk=Chunk(
                        id=r[0], doc_id=r[1], title=r[2], text=r[3], chunk_index=r[4]
                    ),
                    score=float(r[5]),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def count(self) -> int:
        conn = self._conn()
        try:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()

    def clear(self) -> None:
        conn = self._conn()
        try:
            conn.execute("TRUNCATE chunks")
        finally:
            conn.close()
