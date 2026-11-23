"""Retrieval, conservative evidence gating, validated answers, and citations."""

import json
import re
from time import perf_counter

from .models import Answer, Citation
from .pdf_loader import normalize_text

INSUFFICIENT = "I could not find enough evidence in the indexed documents to answer this question."
INJECTION = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions"
    r"|system\s*prompt|developer\s*message|you\s+are\s+now"
    r"|reveal\s+(?:the\s+)?(?:password|secret|token)|send\s+.*credentials",
    re.IGNORECASE,
)


def parse_evidence(raw: str, hits) -> list[Citation]:
    # No free-form model output is rendered, even if it invents a citation.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        items = json.loads(raw)["evidence"]
    except (ValueError, KeyError, TypeError):
        return []
    if not isinstance(items, list) or len(items) > 3:
        return []
    citations, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"source", "quote"}:
            return []
        source, quote = item["source"], item["quote"]
        if (
            type(source) is not int
            or not 1 <= source <= len(hits)
            or not isinstance(quote, str)
            or not 12 <= len(quote) <= 1000
        ):
            return []
        hit = hits[source - 1]
        quote = normalize_text(quote)
        if quote not in normalize_text(hit.chunk.text) or INJECTION.search(quote):
            return []
        if (source, quote) in seen:
            continue
        seen.add((source, quote))
        citations.append(Citation(len(citations) + 1, hit.chunk.id, hit.chunk.filename, hit.chunk.page, quote))
    return citations


class RAGPipeline:
    def __init__(self, store, generator=None):
        self.store, self.generator = store, generator

    def query(self, question: str, top_k: int = 3, min_score: float = 0.35, mode: str = "local") -> Answer:
        if not 0 <= min_score <= 1:
            raise ValueError("Minimum similarity must be between 0 and 1.")
        if mode not in {"local", "retrieval"}:
            raise ValueError("Unknown answer mode.")
        start = perf_counter()
        hits = self.store.search(question, top_k)
        elapsed = (perf_counter() - start) * 1000
        accepted = [h for h in hits if h.score >= min_score and not INJECTION.search(h.chunk.text)]
        if not accepted:
            return Answer(INSUFFICIENT, "insufficient", hits=hits, retrieval_ms=elapsed)
        fallback = Answer(
            "Retrieved passages are shown below. No generated answer is available in this mode.",
            "retrieval-only",
            hits=accepted,
            retrieval_ms=elapsed,
        )
        if mode == "retrieval":
            return fallback
        if self.generator is None:
            fallback.warning = "Local LLM is unavailable. This is retrieval-only output."
            return fallback
        start = perf_counter()
        try:
            raw = self.generator.generate(question, accepted)
        except (RuntimeError, OSError, ValueError) as exc:
            fallback.warning = f"Local generation failed ({type(exc).__name__}); showing retrieval only."
            fallback.generation_ms = (perf_counter() - start) * 1000
            return fallback
        citations = parse_evidence(raw, accepted)
        generation_ms = (perf_counter() - start) * 1000
        if not citations:
            return Answer(
                INSUFFICIENT,
                "insufficient",
                hits=accepted,
                retrieval_ms=elapsed,
                generation_ms=generation_ms,
                warning="The model provided no valid, directly quoted evidence.",
            )
        text = "\n\n".join(f"{c.quote} [{c.number}]" for c in citations)
        return Answer(text, "local-evidence", citations, accepted, elapsed, generation_ms)
