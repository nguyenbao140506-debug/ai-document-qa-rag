import re
from hashlib import sha256

import numpy as np
import pymupdf
import pytest

from ragqa.pdf_loader import extract_pdf


class WordTokenizer:
    """Offline test double; production chunking uses the model's WordPiece tokenizer."""

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]}


class TestEmbedder:
    dimension = 512
    identity = "test-only-hashed-words-v1"
    tokenizer = WordTokenizer()

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        values = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.casefold()):
                index = int.from_bytes(sha256(word.encode()).digest()[:4], "big") % self.dimension
                values[row, index] += 1
            norm = np.linalg.norm(values[row])
            if norm:
                values[row] /= norm
        return values


@pytest.fixture
def embedder():
    return TestEmbedder()


@pytest.fixture
def make_pdf():
    def create(texts, encrypted=False):
        with pymupdf.open() as pdf:
            for text in texts:
                page = pdf.new_page()
                if text:
                    page.insert_textbox((40, 40, 550, 780), text, fontsize=11)
            options = (
                {
                    "encryption": pymupdf.PDF_ENCRYPT_AES_256,
                    "owner_pw": "fixture-owner",
                    "user_pw": "fixture-user",
                }
                if encrypted
                else {}
            )
            return pdf.tobytes(**options)

    return create


@pytest.fixture
def document(make_pdf):
    return extract_pdf(
        make_pdf(
            [
                "Battery telemetry is retained for sixty days.",
                "The emergency stop is checked every morning.",
            ]
        ),
        "lab.pdf",
    )
