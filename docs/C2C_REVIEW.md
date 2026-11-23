# Independent planning and review record

The project used the actual Codex with ChatGPT workflow through the authenticated, read-only `project CV` connector. The canonical workspace name and ID were checked before reading project files. The saved review session is retained in local C2C state.

## Planning

ChatGPT read `docs/TASK_BRIEF.md`, the initial environment audit, directory contents, and Git status through MCP, then returned a concrete implementation plan covering ingestion, chunking, embeddings, FAISS, local answering, Streamlit, tests, evaluation, and evidence. Codex implemented and executed the project. No simulated external PLAN or REVIEW is counted as evidence.

## Review iteration 1

ChatGPT independently inspected staged source files, test cases, evaluation and security records, and listed/read both released execution outputs for iteration 1. It confirmed the actual 77-test baseline, retrieval metrics, and 30/36 expected facts with 8/8 negative abstentions. It returned a fixes-required PLAN, rather than DONE.

Requested fixes and executed responses:

1. **Snapshot vector trust:** a valid checksum and normalized vectors do not prove the embedding/source mapping. Restore now re-embeds bounded source chunks and rejects inconsistent vectors. A regression test uses forged unit vectors and a correctly resealed snapshot. UI, README, and security documentation distinguish this check from original-PDF authentication.
2. **Semantic grounding and adversarial cases:** added tests and actual-model experiments for a real but irrelevant quote, conflicting numeric claims, and an instruction outside the regex filter. A narrow identical-template numeric-conflict guard abstains. All three real-model cases passed; the broader limitations remain explicit.
3. **Evidence consistency:** staged the credential-pattern scan report and reran the scan over updated staged content/history. Benchmark documents are entirely fictional; private uploads, weights, credentials, and indexes remain excluded from Git.
4. **CI and documentation:** added explicit offline test dependencies, the browser-picker verification boundary in the README, and SHA-pinned official GitHub Actions. Local success and remote CI status are recorded separately.

Post-fix local results: **81 pytest tests passed, 91.2% package coverage, three actual-model adversarial cases passed**. The full synthetic QA experiment was rerun on the patched CPU runtime.

## Review iteration 2: implementation accepted

ChatGPT inspected the updated source and evidence through the same workspace connector and read released execution outputs **3 and 4**, both exit code 0. It confirmed snapshot re-embedding and the forged-unit-vector regression, the narrow numeric-conflict guard and semantic/adversarial cases, the staged secret scan, offline dependencies, and the documented browser-picker boundary.

The actual final response returned **`STATE: DONE`**, **`REVIEW RESULT: PASS — Implementation only`**, and **`FINAL DECISION: DONE FOR IMPLEMENTATION`**. It independently matched the final CPU report: 30/36 expected facts, 30/30 correct citations, eight negative refusals, and six answerable misses. It accepted the explicitly documented limitations rather than treating them as general guarantees.

This review did not verify a public push or remote CI. Publication evidence is recorded separately in `PUBLICATION.md` after those actions complete.
