"""Streamlit application. Run: python -m streamlit run app.py."""

import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import streamlit as st

from ragqa.config import ROOT, ChunkConfig
from ragqa.embeddings import SentenceEmbedder
from ragqa.generator import LocalGenerator
from ragqa.pdf_loader import DocumentError, extract_pdf
from ragqa.rag_pipeline import RAGPipeline
from ragqa.vector_store import VectorStore

st.set_page_config(page_title="Evidence Desk | Local Document QA", page_icon="📚", layout="wide")
st.html("""<style>
.stApp {background: #f7f8fc; color: #17223b;}
[data-testid="stSidebar"] {background: #edf1f8;}
h1,h2,h3 {letter-spacing:-.035em;}
.block-container {padding-top:4.5rem; max-width:1400px;}
.eyebrow {font:600 12px monospace;letter-spacing:.18em;color:#4263eb;}
.hero {padding:24px 30px;border-radius:20px;background:#18294b;color:white;margin-bottom:24px;}
.hero h1 {color:white;margin:8px 0;font-size:40px;}
.hero p {color:#d3ddf4;margin-bottom:4px;}
</style>""")


@st.cache_resource(show_spinner="Loading the embedding model…")
def embedding_model():
    return SentenceEmbedder()


@st.cache_resource(show_spinner="Loading the local language model…")
def language_model():
    return LocalGenerator(embedder=embedding_model())


def get_store():
    if "store" not in st.session_state:
        st.session_state.store = VectorStore(embedding_model())
    return st.session_state.store


def clear_history():
    st.session_state.history = []


def show_answer(answer):
    st.caption(
        f"{answer['mode']} · retrieval {answer['retrieval_ms']:.1f} ms · generation {answer['generation_ms']:.1f} ms"
    )
    # Render all uploaded/model text as plain text, never HTML or executable Markdown.
    st.text(answer["text"])
    if answer["warning"]:
        st.warning(answer["warning"])
    if answer["citations"]:
        st.markdown("#### Verified source references")
        for citation in answer["citations"]:
            with st.container(border=True):
                st.text(f"[{citation['number']}] {citation['filename']} · page {citation['page']}")
                st.caption(f"Chunk {citation['chunk_id'][:12]}")
                st.text(citation["quote"])
    for i, hit in enumerate(answer["hits"], 1):
        chunk = hit["chunk"]
        with st.expander(f"Passage {i} · page {chunk['page']} · similarity {hit['score']:.3f}"):
            st.text(chunk["filename"])
            st.text(chunk["text"])


st.html("""<div class="hero"><div class="eyebrow">EVIDENCE DESK / LOCAL RAG</div>
<h1>Your documents. Answers you can trace.</h1>
<p>Ask a question, inspect the evidence, and follow every source back to its page.</p></div>""")

with st.sidebar:
    st.markdown("## Document workspace")
    st.caption("PDF text stays in this session. Model downloads require internet on first use.")
    uploads = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=20,
        help="Up to 20 MB and 200 pages per PDF. Text PDFs only; no OCR.",
    )
    if len(uploads) > 30:
        st.error("Please select at most 30 PDFs at a time.")
    build = st.button(
        "Index uploaded PDFs",
        type="primary",
        disabled=not uploads or len(uploads) > 30,
        use_container_width=True,
    )
    demo = st.button("Load sample documents", use_container_width=True)
    st.divider()
    st.markdown("### Retrieval settings")
    top_k = st.slider("Top-K passages", 1, 10, 3)
    min_score = st.slider(
        "Minimum similarity",
        0.0,
        1.0,
        0.35,
        0.05,
        help="A heuristic similarity filter, not a confidence probability.",
    )
    chunk_choice = st.selectbox("Chunk configuration", ["224 tokens / 48 overlap", "128 tokens / 24 overlap"])
    config = ChunkConfig(128, 24) if chunk_choice.startswith("128") else ChunkConfig(224, 48)
    answer_mode = st.radio(
        "Answer mode",
        ["Local LLM evidence answer", "Retrieval only"],
        help="Local mode selects supporting sentences, then validates every quote.",
    )
    st.caption("Document instructions are treated as untrusted content.")

try:
    if build or demo:
        store = get_store()
        candidates = list(store.documents.values())
        inputs = (
            [(f.name, f.getvalue()) for f in uploads]
            if build
            else [(p.name, p.read_bytes()) for p in sorted((ROOT / "evaluation" / "corpus").glob("*.pdf"))]
        )
        if not inputs:
            st.warning("No sample PDFs found. Run python scripts/build_corpus.py first.")
        progress = st.progress(0, text="Reading PDFs…")
        added = 0
        for i, (name, data) in enumerate(inputs):
            try:
                document = extract_pdf(data, name)
                if document.id not in {d.id for d in candidates}:
                    candidates.append(document)
                    added += 1
                for warning in document.warnings:
                    st.warning(f"{document.filename}: {warning}")
            except DocumentError as exc:
                st.error(f"{Path(name).name}: {exc}")
            progress.progress((i + 1) / len(inputs), text="Extracting and preparing chunks…")
        with st.spinner("Creating embeddings and building the search index…"):
            # A new store makes configuration changes and batch updates atomic.
            if added or (store.config.tokens, store.config.overlap) != (config.tokens, config.overlap):
                replacement = VectorStore(store.embedder, config)
                replacement.replace(candidates)
                st.session_state.store = replacement
        clear_history()
        progress.empty()
        st.success(f"Indexed {added} new document(s). Duplicate content is skipped.")
except (RuntimeError, OSError, ValueError) as exc:
    st.error(f"Indexing could not finish: {exc}")

store = st.session_state.get("store")
left, middle, right = st.columns(3)
left.metric("Indexed documents", len(store.documents) if store else 0)
middle.metric("Searchable passages", len(store.chunks) if store else 0)
right.metric("Indexed pages", sum(d.page_count for d in store.documents.values()) if store else 0)

with st.expander("Manage documents & saved indexes", expanded=bool(store and store.documents)):
    if store:
        for document in list(store.documents.values()):
            info, action = st.columns([5, 1])
            info.text(f"{document.filename} · {document.page_count} page(s) · {document.id[:10]}")
            if action.button("Remove", key=document.id):
                store.remove(document.id)
                clear_history()
                st.rerun()
        if store.chunks:
            st.caption("An index snapshot contains extracted document text. Keep it private.")
            try:
                st.download_button(
                    "Download index snapshot",
                    store.export(),
                    "evidence-desk.json",
                    "application/json",
                )
            except ValueError as exc:
                st.warning(str(exc))
        if (store.config.tokens, store.config.overlap) != (config.tokens, config.overlap):
            st.warning("Chunk selection changed. Re-index documents to apply the selected configuration.")
            if st.button("Apply chunk configuration"):
                with st.spinner("Rebuilding passages…"):
                    replacement = VectorStore(store.embedder, config)
                    replacement.replace(list(store.documents.values()))
                    st.session_state.store = replacement
                clear_history()
                st.rerun()
    st.caption(
        "Restore only snapshots you trust. Filenames and page numbers are not proof of PDF authenticity. "
        "Restoring checks the document text again and may take several seconds."
    )
    saved = st.file_uploader("Restore an index snapshot", type=["json"], key="snapshot")
    if st.button("Restore snapshot", disabled=saved is None):
        try:
            with st.spinner("Validating snapshot and source mappings…"):
                restored = VectorStore.restore(saved.getvalue(), embedding_model())
                st.session_state.store = restored
                clear_history()
            st.rerun()
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(f"Snapshot was rejected: {exc}")

st.markdown("### Ask your documents")
if not store or not store.chunks:
    st.info("Upload text PDFs in the sidebar, or load the sample documents to explore the demo.")
    st.caption("Try: How long are research backups retained? Every answer includes inspectable passages.")
if "history" not in st.session_state:
    st.session_state.history = []
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.text(entry["question"])
    with st.chat_message("assistant"):
        show_answer(entry["answer"])

question = st.chat_input(
    "Ask a question about your indexed PDFs…",
    max_chars=2000,
    disabled=not store or not store.chunks,
)
if question:
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        generator = None
        if answer_mode.startswith("Local"):
            try:
                generator = language_model()
            except (RuntimeError, OSError, ValueError, ImportError) as exc:
                st.warning(f"Local model unavailable ({type(exc).__name__}). Using retrieval-only mode.")
        with st.spinner("Retrieving evidence and checking the answer…"):
            answer = RAGPipeline(store, generator).query(
                question,
                top_k,
                min_score,
                "local" if answer_mode.startswith("Local") else "retrieval",
            )
        answer_data = asdict(answer)
        show_answer(answer_data)
    st.session_state.history.append({"question": question, "answer": answer_data})
    st.session_state.history = st.session_state.history[-10:]

st.divider()
st.caption("Evidence Desk · Sentence Transformers + FAISS · Local inference · Sources are evidence, not instructions")
