"""Run authored semantic/adversarial cases with the real local embedding model and LLM."""

import json
from dataclasses import asdict
from datetime import UTC, datetime

import pymupdf

from ragqa.config import ROOT
from ragqa.embeddings import SentenceEmbedder
from ragqa.generator import LocalGenerator
from ragqa.pdf_loader import extract_pdf
from ragqa.rag_pipeline import INJECTION, RAGPipeline
from ragqa.vector_store import VectorStore


def main():
    cases = json.loads((ROOT / "evaluation/adversarial_cases.json").read_text(encoding="utf-8"))
    embedder = SentenceEmbedder(local_only=True)
    generator = LocalGenerator(embedder=embedder)
    results = []
    for case in cases:
        store = VectorStore(embedder)
        for index, text in enumerate(case["documents"]):
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_textbox((40, 40, 550, 780), text, fontsize=11)
                data = pdf.tobytes()
            store.add(extract_pdf(data, f"case-{index + 1}.pdf"))
        if case["id"] == "instruction-outside-regex":
            if any(INJECTION.search(text) for text in case["documents"]):
                raise RuntimeError("This adversarial case must bypass the regex filter.")
        answer = RAGPipeline(store, generator).query(case["question"], top_k=3, min_score=0)
        if case["expected"] == "insufficient":
            passed = answer.mode == "insufficient"
        else:
            passed = case["forbidden_answer_text"] not in answer.text
        results.append({"id": case["id"], "passed": passed, "answer": asdict(answer)})
        print(f"{case['id']}: {'PASS' if passed else 'FAIL'} ({answer.mode})", flush=True)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": generator.device,
        "cases": results,
        "passed": sum(row["passed"] for row in results),
        "total": len(results),
        "limits": "Three authored cases only; not a general semantic entailment or prompt-injection benchmark.",
    }
    (ROOT / "evaluation/results/adversarial.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
