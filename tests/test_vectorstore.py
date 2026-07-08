from app.embeddings import FakeEmbedder
from app.models import Chunk
from app.vectorstore import InMemoryVectorStore


def _chunk(cid, text, embedder):
    return Chunk(
        id=cid, doc_id="d", title="D", text=text, chunk_index=0,
        embedding=embedder.embed([text])[0],
    )


def test_search_ranks_by_similarity():
    e = FakeEmbedder()
    store = InMemoryVectorStore()
    store.upsert([
        _chunk("a", "vacation and paid time off", e),
        _chunk("b", "password and encryption rules", e),
    ])
    q = e.embed(["how much vacation do I get"])[0]
    results = store.search(q, k=2)
    assert results[0].chunk.id == "a"
    assert results[0].score >= results[1].score


def test_upsert_is_idempotent_on_id():
    e = FakeEmbedder()
    store = InMemoryVectorStore()
    c = _chunk("same-id", "hello", e)
    store.upsert([c])
    store.upsert([c])
    assert store.count() == 1


def test_empty_store_returns_no_results():
    assert InMemoryVectorStore().search([0.0] * 8, k=4) == []
