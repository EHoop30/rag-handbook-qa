import math

from app.embeddings import FakeEmbedder


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def test_fake_embedder_is_deterministic():
    e = FakeEmbedder()
    assert e.embed(["hello world"]) == e.embed(["hello world"])


def test_fake_embedder_vectors_are_normalized():
    e = FakeEmbedder()
    (vec,) = e.embed(["the quick brown fox"])
    assert abs(math.sqrt(sum(v * v for v in vec)) - 1.0) < 1e-6


def test_shared_vocabulary_is_closer_than_unrelated_text():
    e = FakeEmbedder()
    vacation = e.embed(["paid vacation and time off policy"])[0]
    similar = e.embed(["how much vacation time off do I get"])[0]
    unrelated = e.embed(["password encryption security incident"])[0]
    assert _cosine(vacation, similar) > _cosine(vacation, unrelated)
