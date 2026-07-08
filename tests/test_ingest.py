from app.embeddings import FakeEmbedder
from app.ingest import ingest_path
from app.vectorstore import InMemoryVectorStore
from tests.conftest import CORPUS


def test_ingest_indexes_the_corpus():
    store = InMemoryVectorStore()
    stats = ingest_path(CORPUS, FakeEmbedder(), store)
    assert stats["documents"] == 6
    assert stats["chunks"] > 6  # each doc produces at least one chunk
    assert store.count() == stats["chunks"]


def test_reingest_is_idempotent():
    store = InMemoryVectorStore()
    e = FakeEmbedder()
    first = ingest_path(CORPUS, e, store)
    second = ingest_path(CORPUS, e, store)
    # Same content hashes: the second pass upserts over the first, no growth.
    assert store.count() == first["chunks"]
    assert second["total_indexed"] == first["total_indexed"]


def test_every_chunk_gets_an_embedding():
    store = InMemoryVectorStore()
    ingest_path(CORPUS, FakeEmbedder(), store)
    q = FakeEmbedder().embed(["anything"])[0]
    results = store.search(q, k=3)
    assert all(r.chunk.embedding is not None for r in results)
