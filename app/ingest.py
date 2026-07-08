"""Document ingestion: load, chunk, embed, upsert.

Re-running ingestion on an unchanged corpus is a no-op because chunk ids are
content hashes and the store upserts on id. Change a document and only its
chunks get new ids and are re-embedded. Supported inputs: .md, .txt, .pdf.
"""

from __future__ import annotations

from pathlib import Path

from app.chunking import chunk_text
from app.embeddings import Embedder
from app.models import Chunk
from app.vectorstore import VectorStore

_SUPPORTED = {".md", ".txt", ".pdf"}


def _read(path: Path) -> str:
    if path.suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _title(path: Path, text: str) -> str:
    # Use a leading Markdown H1 if present, else the filename stem.
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("-", " ").replace("_", " ").title()


def build_chunks(path: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    text = _read(path)
    doc_id = path.stem
    title = _title(path, text)
    chunks = []
    for i, piece in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
        chunks.append(
            Chunk(
                id=Chunk.make_id(doc_id, i, piece),
                doc_id=doc_id,
                title=title,
                text=piece,
                chunk_index=i,
            )
        )
    return chunks


def ingest_path(
    corpus_dir: str | Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> dict:
    corpus = Path(corpus_dir)
    files = sorted(p for p in corpus.rglob("*") if p.suffix in _SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No supported documents found under {corpus}")

    all_chunks: list[Chunk] = []
    for path in files:
        all_chunks.extend(build_chunks(path, chunk_size, chunk_overlap))

    vectors = embedder.embed([c.text for c in all_chunks])
    for chunk, vec in zip(all_chunks, vectors):
        chunk.embedding = vec

    upserted = store.upsert(all_chunks)
    return {
        "documents": len(files),
        "chunks": upserted,
        "total_indexed": store.count(),
    }
