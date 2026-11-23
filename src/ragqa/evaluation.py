"""Scoring uses authored evidence, never the system's own answer as ground truth."""

from .pdf_loader import normalize_text


def relevant(hit, expected) -> bool:
    return (
        hit.chunk.filename == expected["filename"]
        and hit.chunk.page == expected["page"]
        and normalize_text(expected["evidence"]) in normalize_text(hit.chunk.text)
    )


def retrieval_scores(ranked_hits, expected):
    rank = next((i for i, hit in enumerate(ranked_hits, 1) if relevant(hit, expected)), None)
    return {
        "recall@1": float(rank is not None and rank <= 1),
        "recall@3": float(rank is not None and rank <= 3),
        "recall@5": float(rank is not None and rank <= 5),
        "reciprocal_rank": 1 / rank if rank else 0.0,
    }


def citation_support(citation, expected) -> bool:
    return (
        citation.filename == expected["filename"]
        and citation.page == expected["page"]
        and expected["expected_answer"].casefold() in citation.quote.casefold()
    )
