# Portfolio evidence

Project: **Evidence Desk - AI Document Question-Answering Assistant (RAG)**

Evidence collected on 23-24 November 2026 (Asia/Bangkok). The records below distinguish implemented behaviour, measured results, and limits. External review and publication status are recorded separately; a local success is not a claim of GitHub publication.

## Verified implementation

| Capability | Evidence / code |
| --- | --- |
| PDF validation, page extraction, empty/corrupt/encrypted/scanned handling | [`pdf_loader.py`](../src/ragqa/pdf_loader.py), [`test_pdf.py`](../tests/test_pdf.py) |
| Normalization, page-local WordPiece windows, overlap, stable IDs | [`chunking.py`](../src/ragqa/chunking.py), [`models.py`](../src/ragqa/models.py), [`test_chunking.py`](../tests/test_chunking.py) |
| Real 384D Sentence Transformer embeddings | [`embeddings.py`](../src/ragqa/embeddings.py), real-model integration tests |
| FAISS search, metadata associations, deduplication and document deletion | [`vector_store.py`](../src/ragqa/vector_store.py), [`test_store.py`](../tests/test_store.py) |
| Bounded JSON persistence with integrity/provenance checks and no pickle | Store roundtrip and tamper-rejection tests |
| Local Qwen selection, answerability verification, exact quotes and source citations | [`generator.py`](../src/ragqa/generator.py), [`rag_pipeline.py`](../src/ragqa/rag_pipeline.py), final 44-question experiment |
| Insufficient-evidence response and labelled retrieval-only fallback | Pipeline tests and eight negative evaluation questions |
| Streamlit document management, settings, chat history, progress/errors, source cards | [`app.py`](../app.py), UI smoke test, real running-app screenshots |
| Multi-PDF upload handler | Real Streamlit `UploadedFile` objects containing two actual synthetic PDFs; application button execution and duplicate retry in [`test_integration.py`](../tests/test_integration.py) |

The running app loaded six PDFs, built 36 passages, and answered the Cedar backup question with **45 days**, citing **cedar-research.pdf, page 1**. Supporting passages were available in expanders. Screenshots are actual browser captures, not mockups: [`app-overview.jpg`](../assets/app-overview.jpg) and [`app-answer.jpg`](../assets/app-answer.jpg).

**UI test boundary:** the in-app browser's file-chooser automation failed. Multi-file ingestion was verified at the Streamlit uploaded-file/application-handler boundary, while browser end-to-end querying used the sample-document button through the same extraction/index pipeline. This does not claim a successful automated operating-system file-picker test.

## Automated checks

The full recorded pytest run passed **81 tests**. Coverage of the `ragqa` package was **91.2%** (404/443 statements); this includes offline tests and three real-embedding/application integration tests, not complete coverage of local LLM inference or every UI interaction.

- [`pytest.txt`](../evaluation/results/pytest.txt): actual command output.
- [`pytest.xml`](../evaluation/results/pytest.xml): machine-readable JUnit record.
- [`coverage.json`](../evaluation/results/coverage.json): coverage details.
- [`bandit.json`](../evaluation/results/bandit.json): static-security report.
- [`pip-audit-final.json`](../evaluation/results/pip-audit-final.json): full runtime dependency-closure audit.
- [`secret-scan.json`](../evaluation/results/secret-scan.json): staged/history credential-pattern scan.

An initial sandboxed test invocation failed to initialize the FAISS DLL on Windows. The actual suite passed when executed with the applicable execution permission; no sandbox roots or global settings were changed. Failed-run output is retained separately and is not counted as a passing run.

The GitHub Actions workflow runs offline tests, Ruff formatting/lint, and Bandit without downloading the embedding model or LLM. A workflow file is not evidence that remote CI passed; remote status must be verified after a push.

## Retrieval experiment

Corpus: six original fictional PDFs, 18 pages, 36 independently annotated answerable questions, and eight unanswerable questions. Every answerable item has an expected source filename, one-based page, evidence sentence, and answer phrase. Ground truth is not passed to inference. PDFs and annotations are CC0; hashes are in the corpus manifest.

Final recorded CPU run, Python 3.13.12, Sentence Transformers 5.7.0, Transformers 5.24.0, FAISS 1.16.0, PyMuPDF 1.28.2, PyTorch 2.13.0:

| Configuration | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@5 | p50 / p95 retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 tokens / 24 overlap | 54 | 88.89% | 100% | 100% | 0.9398 | 4.55 / 5.95 ms |
| 224 tokens / 48 overlap | 36 | 100% | 100% | 100% | 1.0000 | 4.18 / 5.58 ms |

Primary evidence: [`final.json`](../evaluation/results/final.json). The initial standalone retrieval experiment is also retained as [`retrieval.json`](../evaluation/results/retrieval.json). Latency includes warm CPU query encoding plus FAISS, three repetitions per answerable question, excluding loading, indexing, and generation. Concurrent background work and hardware affect timing.

## Answer quality, separately measured

The local Qwen model selected an existing sentence identifier, then checked whether that sentence supplied the requested information. The pipeline rendered only validated source text and constructed citations from metadata.

| Diagnostic | Final measured result |
| --- | ---: |
| Expected fact answered on correct source page | 30/36 (83.33%) |
| Correct rendered citations by authored answer/source matching | 30/30 (100%) |
| Rendered quotes present in their cited source chunks | 30/30 (100%) |
| Unanswerable questions correctly refused | 8/8 (100%) |
| Answerable questions refused | 6/36 (16.67%) |

The reported 100% citation score applies to **30 rendered quotes in this corpus**, not all possible questions. Six answerable questions were missed. Exact answer-phrase matching is a diagnostic proxy, not a human entailment or language-quality assessment. The source-support result is largely structural because the validator constructs the answer from existing text.

Earlier experiments are retained: literal quote generation was over-restrictive; sentence-ID selection improved coverage but answered six negative queries incorrectly; adding an explicit answerability check traded some coverage for correct refusal on this small set. See `full-initial.json`, `full-selection-only.json`, and `full.json` for the intermediate GPU trials. The final CPU experiment is `final.json`.

Both the chunk configuration and prompt were developed using this same small synthetic corpus. There is no held-out test set and no claim that these figures generalize to arbitrary documents.

Three additional authored semantic/adversarial cases passed using the actual local models: a genuine but irrelevant quote was refused, identical statements with contradictory numeric values were refused, and an instruction outside the regex filter was ignored while the real retention fact was returned. See [`adversarial.json`](../evaluation/results/adversarial.json). This is evidence for three cases only, not a general adversarial-robustness claim. Regression tests also reject a resealed snapshot containing forged normalized vectors by re-embedding its source chunks.

## Reproduction

Follow the root README installation steps and download the pinned model revisions. Use the Windows/Python 3.13 runtime lock for the tested dependency closure. Then:

```powershell
$env:USE_TF = '0'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
python scripts/build_corpus.py
python -m pytest -s --junitxml=evaluation/results/pytest.xml --cov=ragqa
python scripts/evaluate.py --generation --output evaluation/results/reproduced.json
python -m streamlit run app.py
```

For offline CI-style checks use `python -m pytest -m "not integration"`. The real-model integration tests require the downloaded MiniLM files; the generator experiment requires Qwen as well. No cloud API keys are needed.

## Hardware and operational limits

- An RTX 4060 Laptop GPU with 8,188 MiB VRAM and basic CUDA operations was verified. Initial local Qwen experiments ran on CUDA with the existing PyTorch 2.12 build.
- The final project environment uses patched **PyTorch 2.13 CPU**. Downloads from the CUDA wheel host stalled; the smaller CPU build was installed from PyPI. Original Conda/PyTorch CUDA environments were preserved. The final reported model experiment ran on CPU; it is not a claim of a patched-CUDA benchmark.
- No OCR, table reconstruction, unrestricted synthesis, or production multi-tenant hosting was implemented. Prompt injection is mitigated, not solved. Checksums do not prove source authenticity. See [`SECURITY.md`](SECURITY.md).
- Security reports contain the actual scanner findings. Initial vulnerabilities triggered local dependency upgrades; do not use the initial audit as the final security status.

## Review and publication

The project used an actual ChatGPT planning session through the authenticated, workspace-scoped `project CV` connector. Actual source/diff/test review completed in iteration 2 with **DONE for implementation**. See [`C2C_REVIEW.md`](C2C_REVIEW.md) for the inspected evidence, requested fixes, accepted limitations, and final review outcome.

Target repository: `https://github.com/nguyenbao140506-debug/ai-document-qa-rag`. No publication or remote CI success is asserted by this document until a remote commit hash and verified workflow result are recorded in `PUBLICATION.md`.
