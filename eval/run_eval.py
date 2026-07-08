"""Run the evaluation harness over the golden dataset.

Ingests the corpus into an in-memory store (so the eval is self-contained and
needs no database), then for every gold question measures retrieval quality
(recall@k, MRR) and answer grounding. Prints a summary table and writes a JSON
report to eval/reports/.

Runs fully offline with the default fake providers, which still perform real
bag-of-words retrieval, so the numbers are meaningful. Point it at real
providers by setting EMBEDDING_PROVIDER/LLM_PROVIDER in the environment.

Usage:
    python -m eval.run_eval [--k 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.embeddings import build_embedder
from app.ingest import ingest_path
from app.llm import build_answerer
from app.retrieval import RagService
from app.vectorstore import InMemoryVectorStore
from eval.metrics import answer_contains_score, recall_at_k, reciprocal_rank

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden.jsonl"
CORPUS = ROOT / "corpus"
REPORTS = ROOT / "eval" / "reports"


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def build_service() -> RagService:
    settings = get_settings()
    embedder = build_embedder(settings)
    store = InMemoryVectorStore()
    ingest_path(CORPUS, embedder, store, settings.chunk_size, settings.chunk_overlap)
    return RagService(embedder, store, build_answerer(settings))


def run(k: int) -> dict:
    service = build_service()
    golden = load_golden()

    rows = []
    for item in golden:
        results = service.retrieve(item["question"], k)
        retrieved_docs = [r.chunk.doc_id for r in results]
        recall = recall_at_k(retrieved_docs, item["expected_doc_id"], k)
        rr = reciprocal_rank(retrieved_docs, item["expected_doc_id"])
        answer = service.answerer.answer(item["question"], results)
        grounding = answer_contains_score(answer, item.get("answer_contains", []))
        rows.append(
            {
                "question": item["question"],
                "expected": item["expected_doc_id"],
                "top_doc": retrieved_docs[0] if retrieved_docs else None,
                "recall_at_k": recall,
                "reciprocal_rank": round(rr, 3),
                "grounding": round(grounding, 3),
            }
        )

    n = len(rows)
    summary = {
        "questions": n,
        "k": k,
        "recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 3),
        "mrr": round(sum(r["reciprocal_rank"] for r in rows) / n, 3),
        "answer_grounding": round(sum(r["grounding"] for r in rows) / n, 3),
        "embedding_provider": get_settings().embedding_provider,
        "llm_provider": get_settings().llm_provider,
    }
    return {"summary": summary, "rows": rows}


def print_report(report: dict) -> None:
    s = report["summary"]
    print(f"\nEval over {s['questions']} questions "
          f"(embeddings={s['embedding_provider']}, llm={s['llm_provider']}, k={s['k']})")
    print("-" * 64)
    print(f"  recall@{s['k']:<2}        {s['recall_at_k']:.1%}   (right doc retrieved)")
    print(f"  MRR             {s['mrr']:.3f}   (rank of the right doc)")
    print(f"  answer grounding {s['answer_grounding']:.1%}   (ground-truth fact present)")
    print("-" * 64)
    misses = [r for r in report["rows"] if r["recall_at_k"] < 1.0]
    if misses:
        print(f"  {len(misses)} retrieval miss(es):")
        for r in misses:
            print(f"    - {r['question']}  (wanted {r['expected']}, got {r['top_doc']})")
    else:
        print("  no retrieval misses")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=get_settings().top_k)
    args = parser.parse_args()

    report = run(args.k)
    print_report(report)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
