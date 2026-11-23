# Reproducible evaluation

This is an original synthetic corpus: six fictional institution handbooks, three pages each, 36 answerable questions, and eight unanswerable questions. No personal or third-party copyrighted PDFs are included. The corpus and annotations are offered under CC0-1.0; see `CORPUS_LICENSE.txt`.

`python scripts/build_corpus.py` creates deterministic PDFs with ReportLab invariant mode. `corpus_manifest.json` records their SHA-256 hashes. `questions.json` contains manually authored questions, expected answer phrases, complete evidence sentences, and one-based filename/page references. Expected annotations are used only for scoring; they are never supplied to the retriever or LLM.

Run `python scripts/download_models.py` first. Then run:

```powershell
$env:USE_TF = '0'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
python scripts/evaluate.py --output evaluation/results/retrieval.json
python scripts/evaluate.py --generation --output evaluation/results/full.json
python scripts/evaluate_adversarial.py
```

The experiments compare 128 WordPiece tokens / 24 overlap and 224 / 48, with page-local chunks. Both fit MiniLM's 256-token input limit. FAISS `IndexFlatIP` performs exact search over normalized float32 embeddings. Recall@K is the fraction of answerable queries for which a chunk contains the complete annotated evidence sentence on the correct page of the correct PDF within K results. Each query has one target fact. MRR@5 uses the first matching chunk, or zero if none is present in the five returned results.

Latency is warm query encoding on CPU plus FAISS search. It excludes model loading, index construction, LLM generation, and UI work. Each query is measured three times; raw measurements, p50, and p95 are saved. Hardware, package versions, model revisions, dataset hash, chunks, ranked references, and build times accompany the results. Timings vary by machine and background load.

Generation is a separate experiment using a local Qwen model. The LLM selects exact supporting sentences; a validator constructs filenames/pages and renders only valid quotes. The following metrics have different meanings:

- **Expected-fact answer rate:** answerable queries with at least one quote containing the authored answer phrase on the correct source page, divided by all 36 answerable queries. Abstentions count as failures.
- **Citation correctness:** rendered quotes matching the expected answer phrase and source page, divided by all rendered quotes across all 44 queries. Extra irrelevant quotes count against this metric.
- **Verbatim source support:** quotes present in their cited retrieved chunk with matching filename/page, divided by rendered quotes. This is a structural guarantee of the validator; it does not establish relevance or semantic entailment.
- **Negative abstention:** unanswerable queries returning `insufficient`, divided by eight. Retrieval-only fallback does not count as a correct abstention.
- **Answerable abstention:** answerable queries returning `insufficient`, divided by 36.

These exact-match diagnostics are not human assessments of entailment or fluency. This corpus is small, English-only, fictional, and deliberately clean. The 224/48 default was chosen from this comparison on the same corpus; there is no held-out tuning/test split and no claim of generalization. Similarity 0.35 is a heuristic filter, not calibrated confidence. The application does not perform OCR, table understanding, or multi-document reasoning.

`adversarial_cases.json` adds three separately authored semantic checks: a genuine but irrelevant quotation, identical claims with conflicting numeric values, and a document instruction deliberately outside the regex filter. `evaluate_adversarial.py` runs these with the actual local models and saves each answer in `results/adversarial.json`. Passing three cases is not a broad adversarial robustness claim. The numeric conflict guard covers only identical text templates with changed numbers, not paraphrased or implicit contradictions.
