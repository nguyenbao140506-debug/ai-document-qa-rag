# Environment audit

Observed on 2026-11-23 (Asia/Bangkok). This document records actual checks, not application test results.

## Project identity

- Codex desktop project label: `project CV`.
- Canonical root: `C:\Users\ADMIN\Documents\ChatGPT\project CV`.
- C2C workspace ID: `8f2f6b441d43`; CLI name matches the desktop label exactly.
- Workspace initially contained only `.git`; there was no application code to reuse.
- Git: no commits on `master`, no configured remotes.
- Requested publication target: `nguyenbao140506-debug/ai-document-qa-rag`.
- No remote repository was created, changed, or pushed.

## Python and GPU

| Environment | Python | Relevant installed packages |
| --- | --- | --- |
| `C:\ProgramData\miniconda3\python.exe` | 3.13.12 | PyTorch 2.12.0+cu126, pytest 9.0.3, NumPy 2.4.6, scikit-learn 1.9.0 |
| `C:\Users\ADMIN\.conda\envs\data_science\python.exe` | 3.11.15 | PyTorch 2.12.1+cu126, Streamlit 1.8.0 |
| `C:\Users\ADMIN\.conda\envs\deepface_gpu\python.exe` | 3.10.20 | None of the requested RAG packages found |

- Sentence Transformers, Transformers, FAISS, and PyMuPDF were not installed in any of these environments.
- NVIDIA RTX 4060 Laptop GPU: 8,188 MiB VRAM reported by `nvidia-smi`.
- NVIDIA driver: 592.27; driver reports CUDA support up to 13.1.
- Base PyTorch uses CUDA 12.6 and reports `torch.cuda.is_available() == True`.
- A tensor allocated on CUDA and added to itself returned `[2.0, 2.0, 2.0]` successfully.
- This verifies a basic GPU operation only; model inference has not been tested.
- Windows Python launcher `py -0p` reported no registered Python installations; working Python executables were verified directly.
- `conda env list --json` failed with "The system cannot execute the specified program." Existing named environment interpreters were inspected directly without modifying them.

## Tool availability

- Git is available at `C:\Program Files\Git\cmd\git.exe`.
- Node is available, version v26.12.0 reported by C2C diagnostics.
- GitHub CLI `gh` was not found in PATH, so GitHub CLI authentication could not be checked.
- `cloudflared` was not found in PATH or the standard Program Files locations checked.
- LeanCTX MCP is available. Its command allowlist did not include `py` or `nvidia-smi`; native read-only audit commands were used as allowed by the user's instructions. No LeanCTX settings were changed.

## C2C preflight and authorization boundary

- Ran `c2c workspace -w <canonical-root> --json` and `c2c doctor -w <canonical-root> --no-fix --json` natively.
- No running bridge; no current endpoint URL was returned.
- No saved session or ChatGPT project mapping exists for this workspace.
- Required connector name for a new connection is exactly `project CV`.
- The diagnostic says the C2C state directory is not sandbox-allowlisted. Per the user's policy, sandbox settings and writable roots were not changed. Any necessary state-directory writes must use the applicable approval mechanism.
- Prepared local `.c2c.json` without a name override, and `.c2cignore` to exclude private uploads, indexes, model files, and local environment data.
- No public endpoint was started, no connector was created, and no project access was granted.
- User authorization is required before establishing this project's public authenticated connection and granting ChatGPT read-only project access.
- After authorization: inspect existing connector mappings before creation, preserve exact workspace name, verify `workspace_info` root/ID/name, obtain an actual ChatGPT PLAN, implement and test, then obtain actual ChatGPT REVIEW.

## Current evidence limits

Application implementation, dependency installation, PDF tests, embedding inference, retrieval evaluation, LLM generation, Streamlit screenshots, lint/security checks, commits, and publication have not run. No benchmarks or CV performance claims are established yet.

## Implementation follow-up (2026-11-24)

The preceding sections are the initial audit, not the final project state. The completed local implementation and its measurements are documented in `PORTFOLIO_EVIDENCE.md`.

- Created the project-local `.venv`; existing Conda environments and global settings were preserved.
- Installed and verified real MiniLM embeddings, FAISS retrieval, PDF extraction, and local Qwen inference.
- Initial GPU trials used the existing CUDA PyTorch 2.12. Final security-patched runtime uses PyTorch 2.13 CPU after downloads from the CUDA wheel host stalled.
- Transformers was upgraded to 5.24.0, and vulnerable inherited transitive packages were overridden only inside `.venv`. The 67-package runtime closure passed the final known-vulnerability audit.
- Full pytest suite after review fixes: 81 passed, package statement coverage 91.2%; Ruff, formatting, Bandit and `pip check` passed.
- Original synthetic corpus and real experiment outputs were generated; genuine running-app screenshots were captured.
- Set up and authenticated the exact `project CV` C2C connector after the user's project-specific authorization. An actual ChatGPT PLAN was received, and actual source/test review is in progress in the saved chat. No sandbox roots were changed.
- Located the GitHub account `nguyenbao140506-debug` in Git Credential Manager and verified its identity through the GitHub API without printing or writing its token. Earlier device-code login attempts expired or had a TLS timeout; the existing authorized session avoids creating new credentials.
- Repository creation/push and remote Actions verification are recorded in `PUBLICATION.md` only after those actions actually succeed.
