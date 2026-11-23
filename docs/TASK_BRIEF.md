# AI Document Question-Answering Assistant (RAG)

The user selected the Codex project `project CV` for a complete, verifiable AI/ML internship portfolio project. All user-facing progress updates must be in Vietnamese. README, portfolio evidence, and CV bullets must be in English.

## Required implementation

- Python, Streamlit, Sentence Transformers, FAISS, PyMuPDF, a free local open-source LLM, and pytest.
- Validate one or multiple uploaded PDFs (format and size), extract and normalize text, preserve filenames and page numbers, and handle empty, corrupted, encrypted, and scanned documents gracefully.
- Overlapping chunks with stable identifiers and duplicate-index prevention.
- Configurable Top-K retrieval, correct source metadata, persistent indexes with validated safe reloads.
- Grounded answers with real filename/page citations, supporting passages, insufficient-evidence handling, and protection against instructions in documents.
- If local generation cannot actually run, provide and clearly document a retrieval-only fallback.
- Professional Streamlit UI: upload sidebar, document management, question input, answer and source cards, retrieval settings, progress/errors, basic chat history, and appropriate caching/state handling.

## Required verification

- Reproducible synthetic or redistributable PDF corpus and questions with expected evidence/source references.
- Compare at least two chunking or retrieval configurations.
- Measure Recall@1, Recall@3, Recall@5, MRR, citation correctness, retrieval latency, and answer grounding. Separate retrieval quality from generation quality; document metric definitions and limitations.
- Actual pytest coverage of validation, extraction, empty PDFs, chunk overlap, metadata, embedding dimensions, FAISS retrieval, citations, insufficient evidence, and end-to-end queries. Practical CI tests must not require a large LLM download.
- Run lint and security checks; inspect upload/path handling, persistence risks, and prompt injection.
- Run Streamlit and capture genuine screenshots.
- Save genuine experiment and test output; never invent successful runs or metrics.

## Deliverables

- Maintainable `app.py`, `src/`, `tests/`, `scripts/`, `evaluation/`, `docs/`, `assets/`, dependency setup, license, ignore rules, and GitHub Actions workflow.
- English README: problem, Mermaid architecture, stack, features, installation/run instructions, model/hardware requirements, retrieval method, corpus, measured results, screenshots, limitations, future work.
- `docs/PORTFOLIO_EVIDENCE.md`: implemented features, actual tests/metrics, code references, reproduction, limitations.
- `docs/CV_BULLETS.md`: three concise English CV bullets supported only by verified work.

## Working constraints

- Follow the supplied AGENTS.md: LeanCTX MCP for local context; native C2C commands with explicit verified `-w`; actual ChatGPT PLAN, implementation/testing, actual ChatGPT REVIEW, fixes.
- Obtain project-specific permission before starting a new public endpoint or granting ChatGPT project read access. No automatic sandbox changes, global integration changes, credential disclosure, or edits to unrelated projects.
- Prefer free software and an isolated project environment; reuse appropriate existing packages without changing existing environments.
- Exclude private PDFs, secrets, large weights, and local indexes from Git.
- Target GitHub repository: `https://github.com/nguyenbao140506-debug/ai-document-qa-rag`.
- Explicit publication approval is required before creating a public repository or pushing. Before asking, finish the implementation, tests, evaluations, evidence, review, secret checks, and meaningful local commits.
- Do not claim the project complete while required connection/review or application verification remains blocked.

## Audit

See `docs/ENVIRONMENT_AUDIT.md` for verified local environment facts and the current authorization boundary.
