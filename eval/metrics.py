"""Retrieval and answer-quality metrics for the eval harness.

Retrieval is scored against the expected source document for each question:
- recall@k: was the right document retrieved in the top k?
- reciprocal rank: 1/rank of the right document (0 if not retrieved), whose
  mean over the set is MRR.

Answer quality is scored two ways. The offline check verifies the answer
contains the ground-truth fact tokens, so it runs with no API key. The
optional LLM judge scores faithfulness on a 1 to 5 scale for a richer signal;
it is used only when a real provider is configured.
"""

from __future__ import annotations


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_id: str, k: int) -> float:
    return 1.0 if expected_doc_id in retrieved_doc_ids[:k] else 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], expected_doc_id: str) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id == expected_doc_id:
            return 1.0 / rank
    return 0.0


def answer_contains_score(answer: str, must_contain: list[str]) -> float:
    """Fraction of ground-truth fact tokens present in the answer (offline)."""
    if not must_contain:
        return 1.0
    low = answer.lower()
    hits = sum(1 for tok in must_contain if tok.lower() in low)
    return hits / len(must_contain)


_JUDGE_SYSTEM = (
    "You grade whether a candidate answer is faithful to a reference answer for "
    "a handbook question. Score 1 to 5 where 5 means fully correct and grounded, "
    "3 means partially correct, and 1 means wrong or unsupported. Reply with only "
    "the integer."
)


def llm_judge_faithfulness(client_answer: str, reference: str, question: str, judge) -> int:
    """Score 1 to 5 via an LLM judge. `judge` is a callable(system, user) -> str."""
    user = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Candidate answer: {client_answer}\n\n"
        "Score (1-5):"
    )
    raw = judge(_JUDGE_SYSTEM, user).strip()
    for ch in raw:
        if ch.isdigit():
            return max(1, min(5, int(ch)))
    return 1
