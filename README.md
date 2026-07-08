# RAG Handbook Q&A

A production-shaped retrieval-augmented generation service. It ingests a set of
documents, answers questions over them through a single `/chat` endpoint, and
returns every answer with the source passages it used. It ships with an
evaluation harness that measures retrieval and answer quality against a golden
dataset, so quality is a number you can track, not a vibe.

The example corpus is a fictional company handbook, but the pipeline is
document-agnostic. Point it at policy PDFs, product docs, or a knowledge base.

## Why this is built the way it is

- **Grounded, cited answers.** The LLM is instructed to answer only from
  retrieved context and to say when the answer is not there. Every response
  includes the source passages, so answers are auditable rather than trusted
  blindly.
- **Swappable providers.** Embeddings and answer generation sit behind small
  interfaces. Run OpenAI + Claude in production, or the built-in offline
  providers with no API key at all. Selecting a provider is a config change,
  never a code change.
- **A real vector store and a fast one.** `PgVectorStore` is Postgres +
  pgvector with an HNSW cosine index for production; `InMemoryVectorStore`
  backs the tests and the eval harness so they run in milliseconds with no
  database.
- **Idempotent ingestion.** Chunk ids are content hashes, so re-ingesting an
  unchanged corpus is a no-op and editing a document only re-embeds what
  changed.
- **Measured, not assumed.** The eval harness scores recall@k, MRR, and answer
  grounding against a golden Q&A set.

## Quick start

Runs fully offline with no API keys. The default providers do real bag-of-words
retrieval and extractive answers, so the whole thing works out of the box.

```bash
docker compose up --build
```

Then, in another terminal:

```bash
curl localhost:8000/health

curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question": "How much is the home office stipend?"}'
```

You get a JSON answer plus the source passages it came from. Interactive API
docs are at `localhost:8000/docs`.

## Using real models

Copy `.env.example` to `.env` and switch the providers on:

```
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Then `docker compose up --build` again. OpenAI `text-embedding-3-small` handles
the retrieval and Claude Haiku writes the answers. Everything else is identical.

## The evaluation harness

`eval/golden.jsonl` is a set of questions, each labeled with the document that should answer it and the ground-truth fact.
`python -m eval.run_eval` ingests the corpus, runs every question, and reports:

- **recall@k** is whether the correct document was retrieved in the top k.
- **MRR** captures how highly the correct document was ranked.
- **answer grounding** checks whether the answer actually contains the ground-truth fact.

Running offline with the zero-dependency fake providers:

```
Eval over 18 questions (embeddings=fake, llm=fake, k=4)
  recall@4          88.9%   (right doc retrieved)
  MRR              0.699    (rank of the right doc)
  answer grounding  55.6%   (ground-truth fact present)
```

Those are the numbers for the offline *baseline*. Real embeddings and a real
model score higher; the point of the harness is that you can measure the lift
from any change (a different embedding model, chunk size, or top-k) instead of
guessing. The harness also lists exactly which questions missed, which is how
you debug retrieval rather than staring at it.

## How it works

```
                          POST /chat  { question }
                                 |
                    embed the question (Embedder)
                                 |
                 top-k cosine search (VectorStore)
                                 |
              assemble context + generate (Answerer)
                                 |
              { answer, sources: [ { doc, snippet, score } ] }
```

- `app/ingest.py` loads .md / .txt / .pdf, chunks (`app/chunking.py`), embeds,
  and upserts.
- `app/vectorstore.py` defines the `VectorStore` interface with pgvector and
  in-memory implementations.
- `app/embeddings.py` and `app/llm.py` are the provider abstractions (OpenAI,
  Anthropic, and deterministic offline fakes).
- `app/retrieval.py` is the query path both the API and the eval harness call.
- `app/main.py` is the FastAPI app, wired from environment config.

## Local development

```bash
uv venv && uv pip install -e ".[dev]"
pytest                     # 22 tests, offline, no database needed
python -m eval.run_eval    # the eval harness
```

The test suite uses the in-memory store and fake providers, so it needs no
database and no network. `tests/test_retrieval.py` is a lightweight always-on
version of the eval, asserting the right document comes back for representative
questions.

## Tech

Python 3.11+, FastAPI, PostgreSQL + pgvector (HNSW), OpenAI embeddings, Anthropic
Claude, Docker Compose. Ingestion supports Markdown, text, and PDF.
