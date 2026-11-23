"""Small, explicit defaults shared by UI, experiments, and scripts."""

from dataclasses import dataclass
from pathlib import Path

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATION_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ChunkConfig:
    tokens: int = 224
    overlap: int = 48

    def __post_init__(self):
        if not 16 <= self.tokens <= 224:
            raise ValueError("Chunk size must be 16..224 tokens (MiniLM has a 256-token limit).")
        if not 0 <= self.overlap < self.tokens:
            raise ValueError("Overlap must be non-negative and smaller than chunk size.")


@dataclass(frozen=True)
class UploadLimits:
    max_bytes: int = 20 * 1024 * 1024
    max_pages: int = 200
    max_characters: int = 1_000_000
