"""Immutable data contracts. Page numbers are always one-based."""

from dataclasses import dataclass, field
from hashlib import sha256


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Document:
    id: str
    filename: str
    page_count: int
    pages: tuple[Page, ...]
    warnings: tuple[str, ...] = ()


def chunk_id(document_id: str, page: int, start: int, end: int, text: str) -> str:
    return sha256(f"{document_id}:{page}:{start}:{end}:{text}".encode()).hexdigest()


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    filename: str
    page: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    number: int
    chunk_id: str
    filename: str
    page: int
    quote: str


@dataclass
class Answer:
    text: str
    mode: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    warning: str | None = None
