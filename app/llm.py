"""Answer generation behind one interface.

Given a question and the retrieved context passages, an Answerer returns a
grounded answer. Real providers (Anthropic, OpenAI) are instructed to answer
only from the supplied context and to say when the context does not contain
the answer, which is the main guardrail against hallucination in a RAG system.
The fake provider is extractive and deterministic so the API and eval harness
run offline.
"""

from __future__ import annotations

import re
from typing import Protocol

from app.models import SearchResult

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "do", "does", "how", "what", "when", "i",
    "my", "to", "of", "for", "and", "or", "in", "on", "much", "many", "get",
}

_SYSTEM = (
    "You answer questions about a company's internal handbook using ONLY the "
    "provided context passages. If the answer is not in the context, say you "
    "could not find it in the handbook. Be concise and specific. Do not invent "
    "policy details that are not present in the context."
)


def _format_context(results: list[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] (from \"{r.chunk.title}\")\n{r.chunk.text}")
    return "\n\n".join(blocks)


def _prompt(question: str, results: list[SearchResult]) -> str:
    return (
        f"Context passages:\n\n{_format_context(results)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


class Answerer(Protocol):
    def answer(self, question: str, results: list[SearchResult]) -> str: ...


class FakeAnswerer:
    """Deterministic extractive baseline. No model, no network.

    Selects the sentence from the top retrieved passages with the most overlap
    with the question's content words (a classic extractive-QA baseline), so it
    tends to surface the sentence that actually holds the answer. Good enough to
    run the full request path offline and to give the eval harness a meaningful,
    non-trivial answer-quality signal without an API key.
    """

    def answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "I could not find that in the handbook."
        q_terms = {w for w in _WORD.findall(question.lower()) if w not in _STOP}

        best_sentence = ""
        best_title = results[0].chunk.title
        best_score = -1
        for r in results[:2]:  # consider the two strongest passages
            text = r.chunk.text.replace("\n", " ")
            for sentence in _SENTENCE.split(text):
                terms = set(_WORD.findall(sentence.lower()))
                score = len(q_terms & terms)
                if score > best_score:
                    best_score, best_sentence, best_title = score, sentence.strip(), r.chunk.title

        if best_score <= 0:
            return "I could not find that in the handbook."
        return f'According to "{best_title}": {best_sentence}'


class AnthropicAnswerer:
    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "I could not find that in the handbook."
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=400,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _prompt(question, results)}],
        )
        return "".join(block.text for block in msg.content if block.type == "text").strip()


class OpenAIAnswerer:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "I could not find that in the handbook."
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _prompt(question, results)},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def build_answerer(settings) -> Answerer:
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is unset")
        return AnthropicAnswerer(settings.anthropic_api_key, settings.anthropic_model)
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is unset")
        return OpenAIAnswerer(settings.openai_api_key, settings.openai_chat_model)
    if settings.llm_provider == "fake":
        return FakeAnswerer()
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
