# Evidence-backed CV bullets

- Built a local PDF question-answering assistant with Python, Streamlit, Sentence Transformers, FAISS, and Qwen, preserving filename/page provenance and validating every displayed quotation against retrieved source text.
- Compared two token-based chunking configurations on six synthetic PDFs and 36 annotated questions, improving Recall@1 from 88.9% to 100% and MRR@5 from 0.940 to 1.000 on the authored benchmark.
- Implemented 81 passing pytest checks with 91.2% pipeline coverage, including real embedding retrieval, multi-PDF upload handling, safe index persistence, citation validation, and prompt-injection regression tests; added GitHub Actions quality checks.

Evidence: `PORTFOLIO_EVIDENCE.md`, actual pytest/JUnit/coverage files, and `evaluation/results/final.json`. The retrieval figures are specific to the small synthetic corpus, not production accuracy. These bullets make no claim of cloud deployment or remote CI success.
