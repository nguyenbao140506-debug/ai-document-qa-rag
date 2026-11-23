"""FAISS FlatIP with verified metadata and bounded, pickle-free persistence."""

import json
import re
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import faiss
import numpy as np

from .chunking import split_document
from .config import ChunkConfig, UploadLimits
from .models import Chunk, Document, Hit, Page
from .pdf_loader import safe_filename

MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
MAX_CHUNKS = 3000


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class VectorStore:
    def __init__(self, embedder, config: ChunkConfig | None = None):
        self.embedder, self.config = embedder, config or ChunkConfig()
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.vectors = np.empty((0, embedder.dimension), dtype=np.float32)
        self.index = faiss.IndexFlatIP(embedder.dimension)

    def replace(self, documents: list[Document]) -> None:
        # Build everything before replacing the live state: failed updates are atomic.
        unique = {doc.id: doc for doc in documents}
        if len(unique) > 30:
            raise ValueError("At most 30 distinct documents can be indexed at once.")
        chunks = [c for doc in unique.values() for c in split_document(doc, self.embedder.tokenizer, self.config)]
        if len(chunks) > MAX_CHUNKS:
            raise ValueError(f"Index exceeds the {MAX_CHUNKS}-chunk limit. Use fewer PDFs.")
        vectors = self.embedder.encode([c.text for c in chunks])
        self._set(unique, chunks, vectors)

    def _set(self, documents, chunks, vectors):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), self.embedder.dimension):
            raise ValueError("Embedding dimensions do not match metadata.")
        if not np.isfinite(vectors).all():
            raise ValueError("Vectors must be finite.")
        if len(vectors) and not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise ValueError("Vectors must be normalized for cosine similarity.")
        index = faiss.IndexFlatIP(self.embedder.dimension)
        if len(vectors):
            index.add(vectors)
        self.documents, self.chunks, self.vectors, self.index = documents, chunks, vectors, index

    def add(self, document: Document) -> bool:
        if document.id in self.documents:
            return False
        if len(self.documents) >= 30:
            raise ValueError("At most 30 distinct documents can be indexed at once.")
        new_chunks = split_document(document, self.embedder.tokenizer, self.config)
        if len(self.chunks) + len(new_chunks) > MAX_CHUNKS:
            raise ValueError(f"Index exceeds the {MAX_CHUNKS}-chunk limit. Use fewer PDFs.")
        new_vectors = self.embedder.encode([c.text for c in new_chunks])
        self._set(
            {**self.documents, document.id: document},
            [*self.chunks, *new_chunks],
            np.concatenate([self.vectors, new_vectors]),
        )
        return True

    def remove(self, document_id: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.document_id != document_id]
        self._set(
            {k: d for k, d in self.documents.items() if k != document_id},
            [self.chunks[i] for i in keep],
            self.vectors[keep],
        )

    def search(self, question: str, top_k: int = 3) -> list[Hit]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10:
            raise ValueError("Top-K must be an integer from 1 to 10.")
        if not question.strip() or len(question) > 2000:
            raise ValueError("Question must contain 1..2000 characters.")
        if not self.chunks:
            return []
        query = self.embedder.encode([question])
        if query.shape != (1, self.embedder.dimension) or not np.isfinite(query).all():
            raise ValueError("Invalid query embedding.")
        scores, rows = self.index.search(query, min(top_k, len(self.chunks)))
        return [
            Hit(self.chunks[int(row)], float(score)) for score, row in zip(scores[0], rows[0], strict=True) if row >= 0
        ]

    def export(self) -> bytes:
        payload = {
            "embedding": self.embedder.identity,
            "dimension": self.embedder.dimension,
            "chunking": asdict(self.config),
            "documents": [asdict(d) for d in self.documents.values()],
            "chunks": [asdict(c) for c in self.chunks],
            "vectors": self.vectors.tolist(),
        }
        encoded = _json_bytes(payload)
        result = _json_bytes({"format": "ragqa-index-v1", "sha256": sha256(encoded).hexdigest(), "payload": payload})
        if len(result) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Index snapshot exceeds the 32 MB persistence limit.")
        return result

    def save(self, path: Path) -> None:
        # Path is supplied by trusted application code, never by an uploaded filename.
        data = self.export()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, embedder):
        with path.open("rb") as stream:
            data = stream.read(MAX_SNAPSHOT_BYTES + 1)
        return cls.restore(data, embedder)

    @classmethod
    def restore(cls, data: bytes, embedder):
        if not data or len(data) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Index snapshot is empty or exceeds 32 MB.")
        try:
            envelope = json.loads(data)
            payload = envelope["payload"]
            if envelope["format"] != "ragqa-index-v1":
                raise ValueError("Unsupported index format.")
            if sha256(_json_bytes(payload)).hexdigest() != envelope["sha256"]:
                raise ValueError("Index integrity check failed.")
            if payload["embedding"] != embedder.identity or payload["dimension"] != embedder.dimension:
                raise ValueError("Snapshot embedding model or dimensions do not match.")
            store = cls(embedder, ChunkConfig(**payload["chunking"]))
            raw_docs, raw_chunks = payload["documents"], payload["chunks"]
            if not isinstance(raw_docs, list) or len(raw_docs) > 30:
                raise ValueError("Invalid document count.")
            if not isinstance(raw_chunks, list) or len(raw_chunks) > MAX_CHUNKS:
                raise ValueError("Invalid chunk count.")
            documents = {}
            for raw in raw_docs:
                if not re.fullmatch(r"[0-9a-f]{64}", raw["id"]) or raw["id"] in documents:
                    raise ValueError("Invalid or duplicate document identifier.")
                if safe_filename(raw["filename"]) != raw["filename"]:
                    raise ValueError("Invalid filename in snapshot.")
                count = raw["page_count"]
                if type(count) is not int or not 1 <= count <= UploadLimits().max_pages:
                    raise ValueError("Invalid page count.")
                if not isinstance(raw["pages"], list) or not 0 < len(raw["pages"]) <= count:
                    raise ValueError("Invalid pages.")
                pages = tuple(Page(**p) for p in raw["pages"])
                if (
                    any(
                        type(p.number) is not int
                        or not 1 <= p.number <= count
                        or not isinstance(p.text, str)
                        or not p.text.strip()
                        for p in pages
                    )
                    or len({p.number for p in pages}) != len(pages)
                    or sum(len(p.text) for p in pages) > UploadLimits().max_characters
                ):
                    raise ValueError("Invalid page content or numbering.")
                warnings = raw.get("warnings", [])
                if (
                    not isinstance(warnings, list)
                    or len(warnings) > count
                    or any(not isinstance(w, str) or len(w) > 500 for w in warnings)
                ):
                    raise ValueError("Invalid warnings.")
                doc = Document(raw["id"], raw["filename"], count, pages, tuple(warnings))
                documents[doc.id] = doc
            chunks = [Chunk(**raw) for raw in raw_chunks]
            expected = [c for doc in documents.values() for c in split_document(doc, embedder.tokenizer, store.config)]
            if chunks != expected:
                raise ValueError("Chunk provenance does not match document pages and configuration.")
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
            if not chunks and vectors.size == 0:
                vectors = np.empty((0, embedder.dimension), dtype=np.float32)
            store._set(documents, chunks, vectors)
            # A resealed snapshot can contain arbitrary unit vectors. Verify their
            # source mapping with the active model before accepting the snapshot.
            expected_vectors = embedder.encode([chunk.text for chunk in chunks])
            if not np.allclose(vectors, expected_vectors, rtol=1e-4, atol=1e-4):
                raise ValueError("Snapshot vectors do not match re-embedded source text.")
            return store
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError(f"Cannot restore index: {exc}") from exc
