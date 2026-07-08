"""Deterministic text chunking.

Packs paragraphs greedily into windows of about `chunk_size` characters, with
`chunk_overlap` characters of tail carried into the next window so a fact that
straddles a boundary is still retrievable. Character-based rather than
token-based on purpose: it is deterministic, dependency-free, and good enough
for a document-Q&A corpus. Swapping in a token splitter later is a local change.
"""

from __future__ import annotations

import re

_PARA = re.compile(r"\n\s*\n")


def _split_long_paragraph(para: str, size: int, overlap: int) -> list[str]:
    out: list[str] = []
    start = 0
    while start < len(para):
        end = start + size
        out.append(para[start:end].strip())
        if end >= len(para):
            break
        start = end - overlap
    return [p for p in out if p]


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> list[str]:
    paragraphs = [p.strip() for p in _PARA.split(text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_paragraph(para, chunk_size, chunk_overlap))
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            # Start the next window with an overlapping tail of the previous one.
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)
    return chunks
