"""Scan staged content and every reachable Git-history blob for credential patterns.

This is a transparent heuristic check, not a guarantee that no secret exists.
Only match locations and pattern labels are printed; matching values are never logged.
"""

import json
import re
import subprocess

from ragqa.config import ROOT

PATTERNS = {
    "github-token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "huggingface-token": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "openai-token": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{30,}"),
    "aws-access-key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "literal-api-secret": re.compile(
        rb"""(?i)(?:api_key|access_token|client_secret)\s*[=:]\s*["'][A-Za-z0-9_/-]{24,}["']"""
    ),
}
FORBIDDEN = re.compile(r"(^|/)(?:\.env(?:\..*)?|secrets\.toml|credentials\.json)$")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)  # nosec B603 B607


def main():
    staged = git("ls-files", "--stage", "-z").split(b"\x00")
    blobs = {}
    forbidden = []
    for entry in staged:
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        name = path.decode("utf-8", errors="replace")
        if FORBIDDEN.search(name) and name != ".env.example":
            forbidden.append(name)
        blobs[header.split()[1].decode()] = name
    staged_count = len(blobs)
    history = git("rev-list", "--objects", "--all").decode("utf-8", errors="replace").splitlines()
    history_blobs = 0
    for line in history:
        oid, _, name = line.partition(" ")
        if git("cat-file", "-t", oid).strip() == b"blob":
            history_blobs += 1
            blobs.setdefault(oid, name)
    findings = []
    for oid, name in blobs.items():
        content = git("cat-file", "blob", oid)
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append({"path": name, "pattern": label, "blob": oid})
    report = {
        "staged_blobs": staged_count,
        "reachable_history_blobs": history_blobs,
        "unique_blobs_scanned": len(blobs),
        "forbidden_paths": forbidden,
        "findings": findings,
        "scope": "All staged content and reachable Git-history blobs; known credential patterns only.",
    }
    destination = ROOT / "evaluation" / "results" / "secret-scan.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(findings or forbidden))


if __name__ == "__main__":
    main()
