"""Run real retrieval experiments and an optional separately scored local LLM run."""

import argparse
import importlib.metadata
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from time import perf_counter

import numpy as np

from ragqa.config import (
    EMBEDDING_REVISION,
    GENERATION_REVISION,
    ROOT,
    ChunkConfig,
)
from ragqa.embeddings import SentenceEmbedder
from ragqa.evaluation import citation_support, retrieval_scores
from ragqa.generator import LocalGenerator
from ragqa.pdf_loader import extract_pdf
from ragqa.rag_pipeline import RAGPipeline
from ragqa.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--output", default="evaluation/results/retrieval.json")
    args = parser.parse_args()
    questions_path = ROOT / "evaluation" / "questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    documents = [
        extract_pdf(path.read_bytes(), path.name) for path in sorted((ROOT / "evaluation" / "corpus").glob("*.pdf"))
    ]
    embedder = SentenceEmbedder(local_only=True)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "embedding_revision": EMBEDDING_REVISION,
        "generation_revision": GENERATION_REVISION,
        "questions_sha256": sha256(questions_path.read_bytes()).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "sentence-transformers", "transformers", "faiss-cpu", "PyMuPDF"]
        },
        "documents": len(documents),
        "pages": sum(d.page_count for d in documents),
        "answerable_questions": sum(q["answerable"] for q in questions),
        "negative_questions": sum(not q["answerable"] for q in questions),
        "latency_scope": (
            "Warm query embedding on CPU plus FAISS search; 3 repetitions per question; "
            "excludes model loading and indexing."
        ),
        "configurations": [],
    }
    stores = []
    for config in [ChunkConfig(128, 24), ChunkConfig(224, 48)]:
        store = VectorStore(embedder, config)
        start = perf_counter()
        store.replace(documents)
        build_ms = (perf_counter() - start) * 1000
        store.search(questions[0]["question"], 5)  # warm up; excluded from measured latency
        rows, latencies = [], []
        for item in questions:
            if not item["answerable"]:
                continue
            for _ in range(3):
                start = perf_counter()
                hits = store.search(item["question"], 5)
                latencies.append((perf_counter() - start) * 1000)
            row = {
                "id": item["id"],
                **retrieval_scores(hits, item),
                "retrieved": [
                    {
                        "chunk_id": h.chunk.id,
                        "filename": h.chunk.filename,
                        "page": h.chunk.page,
                        "score": h.score,
                    }
                    for h in hits
                ],
            }
            rows.append(row)
        metrics = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ["recall@1", "recall@3", "recall@5", "reciprocal_rank"]
        }
        metrics["mrr@5"] = metrics.pop("reciprocal_rank")
        metrics.update(
            {
                "retrieval_p50_ms": float(np.percentile(latencies, 50)),
                "retrieval_p95_ms": float(np.percentile(latencies, 95)),
                "build_ms": build_ms,
            }
        )
        result = {
            "chunking": asdict(config),
            "chunks": len(store.chunks),
            "metrics": metrics,
            "questions": rows,
            "latency_samples_ms": latencies,
        }
        report["configurations"].append(result)
        stores.append(store)
        print(json.dumps({"chunking": asdict(config), "metrics": metrics}), flush=True)
    if args.generation:
        generator = LocalGenerator(embedder=embedder)
        generation_store = stores[1]  # The 224/48 configuration selected in the exploratory comparison.
        pipeline = RAGPipeline(generation_store, generator)
        rows, valid_quotes, citation_total, correct_citations, correct_answers = [], 0, 0, 0, 0
        for item in questions:
            answer = pipeline.query(item["question"], top_k=3, min_score=0.35)
            correct = False
            for citation in answer.citations:
                citation_total += 1
                chunk = next(c for c in generation_store.chunks if c.id == citation.chunk_id)
                valid_quotes += int(
                    citation.quote in chunk.text and citation.filename == chunk.filename and citation.page == chunk.page
                )
                supported = item["answerable"] and citation_support(citation, item)
                correct_citations += int(supported)
                correct |= bool(supported)
            correct_answers += int(correct)
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answerable": item["answerable"],
                    "expected_answer": item.get("expected_answer"),
                    "expected_source": {"filename": item.get("filename"), "page": item.get("page")},
                    "expected_fact_supported": correct,
                    "answer": asdict(answer),
                }
            )
            print(
                f"{item['id']}: {answer.mode}; expected fact={correct}; generation={answer.generation_ms:.1f} ms",
                flush=True,
            )
        positive_count = report["answerable_questions"]
        negatives = [r for r in rows if not r["answerable"]]
        report["generation"] = {
            "model": generator.identity,
            "device": generator.device,
            "configuration": asdict(generation_store.config),
            "min_score": 0.35,
            "top_k": 3,
            "metrics": {
                "expected_fact_answer_rate": correct_answers / positive_count,
                "citation_correctness": correct_citations / citation_total if citation_total else None,
                "verbatim_source_support_rate": valid_quotes / citation_total if citation_total else None,
                "citation_count": citation_total,
                "negative_abstention_rate": sum(r["answer"]["mode"] == "insufficient" for r in negatives)
                / len(negatives),
                "answerable_abstention_rate": sum(
                    r["answer"]["mode"] == "insufficient" for r in rows if r["answerable"]
                )
                / positive_count,
            },
            "scoring_limitations": (
                "Exact expected-answer and source matching is a diagnostic proxy; it is not a human "
                "semantic entailment or language-quality evaluation. Verbatim support is structural, "
                "not evidence relevance. Configuration was selected using this same small synthetic corpus."
            ),
            "questions": rows,
        }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved measured report: {output}", flush=True)


if __name__ == "__main__":
    main()
