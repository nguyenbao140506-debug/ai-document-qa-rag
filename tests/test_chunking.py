import pytest

from ragqa.chunking import split_document
from ragqa.config import ChunkConfig
from ragqa.models import Document, Page


def test_token_overlap_stability_and_provenance(embedder):
    text = " ".join(f"word{i}" for i in range(55))
    doc = Document("a" * 64, "test.pdf", 2, (Page(1, text), Page(2, "Another page.")))
    config = ChunkConfig(20, 5)
    chunks = split_document(doc, embedder.tokenizer, config)
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]
    assert len(chunks[0].text.split()) == 20
    assert chunks == split_document(doc, embedder.tokenizer, config)
    assert len({c.id for c in chunks}) == len(chunks)
    for chunk in chunks:
        page = next(p for p in doc.pages if p.number == chunk.page)
        assert chunk.text == page.text[chunk.start : chunk.end]
        assert chunk.document_id == doc.id and chunk.filename == doc.filename
    assert chunks[-1].page == 2


@pytest.mark.parametrize("tokens,overlap", [(0, 0), (225, 10), (16, 16), (20, -1)])
def test_invalid_configuration(tokens, overlap):
    with pytest.raises(ValueError):
        ChunkConfig(tokens, overlap)


def test_boundary_chunk_no_duplicate_tail(embedder):
    doc = Document("a" * 64, "a.pdf", 1, (Page(1, " ".join(["word"] * 32)),))
    chunks = split_document(doc, embedder.tokenizer, ChunkConfig(16, 0))
    assert len(chunks) == 2
