"""Token-aware, page-local overlapping chunks with stable content identifiers."""

from .config import ChunkConfig
from .models import Chunk, Document, chunk_id


def split_document(document: Document, tokenizer, config: ChunkConfig) -> list[Chunk]:
    chunks = []
    for page in document.pages:
        offsets = tokenizer(
            page.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )["offset_mapping"]
        step = config.tokens - config.overlap
        for token_start in range(0, len(offsets), step):
            window = offsets[token_start : token_start + config.tokens]
            start, end = window[0][0], window[-1][1]
            text = page.text[start:end]
            if text.strip():
                chunks.append(
                    Chunk(
                        chunk_id(document.id, page.number, start, end, text),
                        document.id,
                        document.filename,
                        page.number,
                        start,
                        end,
                        text,
                    )
                )
            if token_start + config.tokens >= len(offsets):
                break
    return chunks
