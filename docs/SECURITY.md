# Security review and operational scope

Evidence Desk is a localhost, single-user portfolio application. It has no shell tools, external actions, paid inference APIs, or credential fields. Uploaded document content is never treated as an instruction to operate the computer.

## Controls verified in code and tests

- PDF names are reduced to safe basenames and used as display metadata, not filesystem destinations. Format, magic bytes, size, page count, extracted-character count, document count, and chunk count have explicit limits.
- Empty and image-only PDFs produce an OCR-required message; corrupted and encrypted PDFs produce safe errors. No embedded attachments, scripts, links, or document actions are executed.
- Index updates build replacement state before publishing it. Duplicate document bytes are identified by SHA-256. Deleting a document removes its vectors and metadata together. Tests cover incorrect dimensions and atomic failure.
- Persistence uses bounded JSON, with no pickle or native FAISS file loading. A restore verifies the envelope checksum, model identity, vector dimensions and norms, valid document metadata, page ranges, unique IDs, and exactly regenerated chunk provenance. It then re-embeds the source chunks and rejects stored vectors that disagree beyond float32 tolerance, including a resealed snapshot containing forged unit vectors. This costs a bounded embedding pass. The checksum alone protects against accidental damage, not a malicious author who recalculates it; metadata and original PDF authenticity remain unverified.
- The local model uses pinned download revisions, safetensors, `trust_remote_code=False`, and local-only generation loads. User uploads cannot select a model path or provide a model checkpoint.
- Prompt context is explicitly untrusted. A suspicious-instruction filter removes matching passages. The model selects only existing sentence identifiers and separately checks answerability. The final validator accepts only exact source quotes and constructs filename/page references itself. Identical sentence templates with different numeric values trigger abstention; this narrow guard is not general contradiction detection.
- Uploaded text, filenames, and model output are rendered as plain text. The UI's HTML/CSS is a constant authored string; no uploaded text is interpolated into it.
- Index state and conversation history are per Streamlit session. Shared cached models are guarded by locks. History is limited to ten turns and is cleared after document/index changes.
- Server configuration binds to `127.0.0.1`; usage telemetry is disabled. Models, local indexes, private uploads, secrets, and virtual environments are ignored by Git. The separate C2C connection is development infrastructure, not part of the application.

## Checks and findings

Actual test, Bandit, dependency-audit, and staged/history scan records are under `evaluation/results/`. The initial audit identified advisories in Transformers 4.57.6 and PyTorch 2.12.0, prompting a project-local upgrade. Consult the final audit record for the actual versions and unresolved findings; an initial clean static scan alone does not establish dependency safety.

`scripts/check_secrets.py` scans all staged and reachable historical Git blobs for documented credential patterns without printing matching values. It also flags sensitive filenames. This heuristic is not a substitute for key rotation if a real secret is ever exposed.

## Remaining risks

- PDF parsers are native software. In-process parsing and limits do not give a strict CPU/memory timeout or protect against every parser vulnerability. Public multi-user hosting would require isolated workers and resource quotas.
- Prompt injection and document poisoning are not solved. Source constraints prevent arbitrary generated text and invented source metadata, but a poisoned document may itself contain false facts or persuasive instructions. A cited quote is evidence of what a document says, not of truth.
- Answerability is heuristic and model-dependent. Supported answers can be missed; an irrelevant quote may pass checks. The synthetic evaluation reports relevance, quotation integrity, and abstention separately.
- A JSON snapshot contains plaintext extracted content and can assert metadata for its own documents. It is not a signed archive or proof of original PDF authenticity. Only restore snapshots from trusted sources.
- Session cleanup depends on Streamlit's lifecycle. Restart the process to clear memory; keep downloaded snapshots private. Sensitive uploads should not be used in shared demonstrations.
- No authentication, authorization, secure multi-tenant storage, encryption at rest, or production deployment controls are provided. Do not expose the app beyond localhost without a separate review.
