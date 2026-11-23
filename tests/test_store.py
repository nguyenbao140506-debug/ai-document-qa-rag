import json
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from ragqa.models import Document, Page
from ragqa.vector_store import VectorStore


def reseal(envelope):
    payload = json.dumps(
        envelope["payload"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    envelope["sha256"] = sha256(payload).hexdigest()
    return json.dumps(envelope).encode()


def test_faiss_search_maps_correct_document(embedder, document):
    store = VectorStore(embedder)
    store.add(document)
    assert store.index.ntotal == len(store.chunks)
    assert store.vectors.shape == (len(store.chunks), embedder.dimension)
    assert np.allclose(np.linalg.norm(store.vectors, axis=1), 1)
    hit = store.search("emergency stop checked", 1)[0]
    assert hit.chunk.filename == "lab.pdf" and hit.chunk.page == 2
    assert "emergency" in hit.chunk.text


def test_deduplication_skips_embedding(embedder, document):
    store = VectorStore(embedder)
    assert store.add(document)
    calls = len(embedder.calls)
    assert not store.add(replace(document, filename="renamed.pdf"))
    assert len(embedder.calls) == calls and len(store.documents) == 1


def test_remove_keeps_other_metadata_without_embedding(embedder, document):
    store = VectorStore(embedder)
    store.add(document)
    other = Document("b" * 64, "other.pdf", 1, (Page(1, "Solar panel inspections happen weekly."),))
    store.add(other)
    calls = len(embedder.calls)
    store.remove(document.id)
    assert len(embedder.calls) == calls
    assert store.search("Solar panel", 5)[0].chunk.document_id == other.id
    assert all(c.filename == "other.pdf" for c in store.chunks)


def test_persistence_roundtrip_and_empty_store(embedder, document, tmp_path):
    store = VectorStore(embedder)
    store.add(document)
    path = tmp_path / "index.json"
    store.save(path)
    loaded = VectorStore.load(path, embedder)
    assert loaded.chunks == store.chunks
    assert np.array_equal(loaded.vectors, store.vectors)
    assert loaded.search("emergency stop", 1) == store.search("emergency stop", 1)
    empty = VectorStore.restore(VectorStore(embedder).export(), embedder)
    assert empty.search("anything") == []


@pytest.mark.parametrize(
    "mutation",
    [
        "integrity",
        "dimension",
        "model",
        "page",
        "chunk",
        "vector",
        "filename",
        "duplicate_document",
        "format",
    ],
)
def test_corrupt_or_mismatched_snapshots_rejected(embedder, document, mutation):
    store = VectorStore(embedder)
    store.add(document)
    envelope = json.loads(store.export())
    payload = envelope["payload"]
    if mutation == "integrity":
        envelope["sha256"] = "0" * 64
    elif mutation == "dimension":
        payload["dimension"] += 1
    elif mutation == "model":
        payload["embedding"] = "other-model"
    elif mutation == "page":
        payload["documents"][0]["pages"][0]["number"] = 999
    elif mutation == "chunk":
        payload["chunks"][0]["filename"] = "forged.pdf"
    elif mutation == "vector":
        payload["vectors"][0] = [0] * embedder.dimension
    elif mutation == "filename":
        payload["documents"][0]["filename"] = "../../secret.pdf"
    elif mutation == "duplicate_document":
        payload["documents"].append(payload["documents"][0])
    else:
        envelope["format"] = "arbitrary-pickle"
    data = json.dumps(envelope).encode() if mutation == "integrity" else reseal(envelope)
    with pytest.raises(ValueError):
        VectorStore.restore(data, embedder)


@pytest.mark.parametrize("data", [b"not json", b"[]", b"{}", b"\x80\x04pickle", b"null"])
def test_non_snapshots_rejected(embedder, data):
    with pytest.raises(ValueError):
        VectorStore.restore(data, embedder)


def test_failed_update_does_not_change_index(embedder, document, monkeypatch):
    store = VectorStore(embedder)
    store.add(document)
    previous = store.export()
    monkeypatch.setattr(embedder, "encode", lambda texts: np.zeros((1, 3)))
    with pytest.raises(ValueError):
        store.replace([document])
    assert store.export() == previous


def test_resealed_snapshot_with_forged_normalized_vectors_rejected(embedder, document):
    store = VectorStore(embedder)
    store.add(document)
    envelope = json.loads(store.export())
    forged = np.roll(np.asarray(envelope["payload"]["vectors"], dtype=np.float32), 1, axis=1)
    assert np.allclose(np.linalg.norm(forged, axis=1), 1)
    envelope["payload"]["vectors"] = forged.tolist()
    with pytest.raises(ValueError, match="re-embedded"):
        VectorStore.restore(reseal(envelope), embedder)


@pytest.mark.parametrize("top_k", [0, 11, -1, True, 1.5])
def test_invalid_topk(embedder, top_k):
    with pytest.raises(ValueError):
        VectorStore(embedder).search("question", top_k)


@pytest.mark.parametrize("question", ["", "  ", "a" * 2001])
def test_invalid_questions(embedder, question):
    with pytest.raises(ValueError):
        VectorStore(embedder).search(question)
