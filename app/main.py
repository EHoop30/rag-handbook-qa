"""FastAPI application.

`create_app` is a factory. In production it wires real providers and pgvector
from environment config on startup; in tests it takes an already-built
`RagService` (with fakes and an in-memory store), so importing this module
never touches a database or a network.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import Settings, get_settings
from app.embeddings import build_embedder
from app.llm import build_answerer
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retrieval import RagService
from app.vectorstore import PgVectorStore


def build_service(settings: Settings) -> RagService:
    embedder = build_embedder(settings)
    store = PgVectorStore(settings.database_url, dim=embedder.dim)
    answerer = build_answerer(settings)
    return RagService(embedder, store, answerer)


def create_app(service: RagService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            app.state.service = build_service(get_settings())
        yield

    app = FastAPI(title="RAG Handbook Q&A", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        svc: RagService = app.state.service
        return HealthResponse(status="ok", chunks_indexed=svc.store.count())

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        svc: RagService = app.state.service
        k = req.k or get_settings().top_k
        try:
            return svc.chat(req.question, k)
        except Exception as exc:  # noqa: BLE001 - surface a clean 500, log the detail
            raise HTTPException(status_code=500, detail=f"chat failed: {exc}") from exc

    return app


# For `uvicorn app.main:app`. Service is built lazily on startup.
app = create_app()
