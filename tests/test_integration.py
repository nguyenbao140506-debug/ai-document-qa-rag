import json

import numpy as np
import pytest
import streamlit as st
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

from ragqa.config import ROOT, ChunkConfig
from ragqa.embeddings import SentenceEmbedder
from ragqa.pdf_loader import extract_pdf
from ragqa.vector_store import VectorStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model():
    return SentenceEmbedder(local_only=True)


def test_real_sentence_transformer_dimensions_and_similarity(model):
    vectors = model.encode(
        [
            "A kitten sleeps on a sofa.",
            "A cat rests on the couch.",
            "Industrial pumps require periodic lubrication.",
        ]
    )
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
    assert model.encode([]).shape == (0, 384)


def test_real_pdf_faiss_and_persistence(model, tmp_path):
    path = ROOT / "evaluation" / "corpus" / "cedar-research.pdf"
    store = VectorStore(model, ChunkConfig(128, 24))
    store.add(extract_pdf(path.read_bytes(), path.name))
    hits = store.search("How long are Cedar research backups retained?", 3)
    assert any("45 days" in hit.chunk.text and hit.chunk.page == 1 for hit in hits)
    assert all(len(model.tokenizer(c.text)["input_ids"]) <= 256 for c in store.chunks)
    store.save(tmp_path / "real.json")
    restored = VectorStore.load(tmp_path / "real.json", model)
    assert restored.search("Cedar backups", 1) == store.search("Cedar backups", 1)
    assert len(json.loads(store.export())["payload"]["vectors"][0]) == 384


def test_streamlit_multiple_uploaded_file_handler(monkeypatch):
    """Real Streamlit uploaded-file objects and PDF bytes; browser chooser is separate."""
    paths = [ROOT / "evaluation/corpus/cedar-research.pdf", ROOT / "evaluation/corpus/atlas-robotics.pdf"]
    uploaded = [
        UploadedFile(UploadedFileRec(str(i), p.name, "application/pdf", p.read_bytes()), FileURLs())
        for i, p in enumerate(paths)
    ]
    real_uploader = st.file_uploader

    def uploader(label, *args, **kwargs):
        return uploaded if label == "Upload PDF documents" else real_uploader(label, *args, **kwargs)

    monkeypatch.setattr(st, "file_uploader", uploader)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    next(button for button in app.button if button.label == "Index uploaded PDFs").click().run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == "2" and int(app.metric[1].value) > 0
    next(button for button in app.button if button.label == "Index uploaded PDFs").click().run(timeout=30)
    assert app.metric[0].value == "2"
    assert "0 new document" in app.success[0].value
