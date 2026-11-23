"""Lazy model loading and normalized embeddings; no model download on import."""

import os
from threading import RLock

import numpy as np

from .config import EMBEDDING_MODEL, EMBEDDING_REVISION, ROOT


class SentenceEmbedder:
    def __init__(self, device: str = "cpu", local_only: bool = False):
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from sentence_transformers import SentenceTransformer

        local = ROOT / "models" / "minilm"
        source = str(local) if (local / "model.safetensors").is_file() else EMBEDDING_MODEL
        self._lock = RLock()
        self.model = SentenceTransformer(
            source,
            revision=EMBEDDING_REVISION if source == EMBEDDING_MODEL else None,
            device=device,
            trust_remote_code=False,
            local_files_only=local_only,
        )
        self.tokenizer = self.model.tokenizer
        self.dimension = self.model.get_embedding_dimension()
        self.identity = f"{EMBEDDING_MODEL}@{EMBEDDING_REVISION}:normalized"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        with self._lock:
            vectors = self.model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape != (len(texts), self.dimension) or not np.isfinite(vectors).all():
            raise ValueError("Embedding model returned invalid vectors.")
        return vectors
