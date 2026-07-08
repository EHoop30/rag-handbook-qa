"""The RAG query path: embed the question, retrieve, answer, cite.

`RagService` is the seam the API and the eval harness both call. It holds the
three swappable pieces (embedder, vector store, answerer) and turns a question
into a grounded answer plus the sources it used, so every answer is auditable.
"""

from __future__ import annotations

from app.embeddings import Embedder, embed_query
from app.llm import Answerer
from app.models import ChatResponse, Source
from app.vectorstore import VectorStore

_SNIPPET_LEN = 240


class RagService:
    def __init__(
        self, embedder: Embedder, store: VectorStore, answerer: Answerer
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.answerer = answerer

    def retrieve(self, question: str, k: int):
        q_vec = embed_query(self.embedder, question)
        return self.store.search(q_vec, k)

    def chat(self, question: str, k: int) -> ChatResponse:
        results = self.retrieve(question, k)
        answer = self.answerer.answer(question, results)
        sources = [
            Source(
                doc_id=r.chunk.doc_id,
                title=r.chunk.title,
                snippet=r.chunk.text[:_SNIPPET_LEN].strip(),
                score=round(r.score, 4),
            )
            for r in results
        ]
        return ChatResponse(question=question, answer=answer, sources=sources)
