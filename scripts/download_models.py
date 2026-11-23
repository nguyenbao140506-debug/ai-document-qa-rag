"""Download only required safetensors/tokenizer files at immutable model revisions."""

import json
import os

from ragqa.config import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    GENERATION_MODEL,
    GENERATION_REVISION,
    ROOT,
)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main():
    from huggingface_hub import snapshot_download

    models = [
        (EMBEDDING_MODEL, EMBEDDING_REVISION, "minilm"),
        (GENERATION_MODEL, GENERATION_REVISION, "qwen"),
    ]
    patterns = [
        "*.json",
        "*.safetensors",
        "*.txt",
        "*.model",
        "*.jinja",
        "1_Pooling/*",
        "README.md",
        "LICENSE",
        "merges.txt",
        "vocab.json",
    ]
    manifest = []
    for model, revision, folder in models:
        print(f"Downloading {model} at {revision}", flush=True)
        snapshot_download(
            model,
            revision=revision,
            allow_patterns=patterns,
            local_dir=str(ROOT / "models" / folder),
            max_workers=3,
        )
        manifest.append({"model": model, "revision": revision, "directory": folder})
    (ROOT / "models" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Models ready. Weights remain excluded from Git.")


if __name__ == "__main__":
    main()
