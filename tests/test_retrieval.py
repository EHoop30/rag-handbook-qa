"""Retrieval quality on the real corpus, using offline fake providers.

These assert that the right source document comes back for representative
questions. They are a lightweight, always-on version of the eval harness.
"""

import pytest


@pytest.mark.parametrize(
    "question,expected_doc",
    [
        ("How many vacation days do I get?", "pto-and-leave"),
        ("What are the core hours for remote work?", "remote-work-policy"),
        ("What is the mileage reimbursement rate?", "expenses-and-travel"),
        ("What is the minimum password length?", "security-and-devices"),
        ("How much does the company contribute to my retirement?", "benefits-and-401k"),
    ],
)
def test_retrieves_expected_document(service, question, expected_doc):
    results = service.retrieve(question, k=3)
    top_docs = [r.chunk.doc_id for r in results]
    assert expected_doc in top_docs


def test_chat_returns_answer_and_cited_sources(service):
    resp = service.chat("How much is the home office stipend?", k=4)
    assert resp.answer
    assert resp.sources
    assert all(0 <= s.score <= 1.0001 for s in resp.sources)
    # The stipend fact lives in the remote-work policy; it should be cited.
    assert any(s.doc_id == "remote-work-policy" for s in resp.sources)
