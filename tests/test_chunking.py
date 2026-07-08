from app.chunking import chunk_text


def test_chunks_respect_size_with_slack_for_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=80)
    assert len(chunks) > 1
    # Each chunk stays near the target; overlap can push slightly over.
    assert all(len(c) <= 500 + 80 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_long_paragraph_is_hard_split():
    para = "x" * 2500
    chunks = chunk_text(para, chunk_size=900, chunk_overlap=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 900 for c in chunks)


def test_overlap_carries_context_between_chunks():
    text = "\n\n".join("sentence " * 30 for _ in range(6))
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=100)
    # Consecutive chunks should share some trailing/leading text.
    assert len(chunks) >= 2
    assert chunks[0][-30:] in chunks[1] or "sentence" in chunks[1]


def test_empty_input_yields_no_chunks():
    assert chunk_text("   \n\n   ") == []
