"""In-memory PDF validation and extraction; upload names never become paths."""

import re
import unicodedata
from hashlib import sha256
from pathlib import PureWindowsPath

import pymupdf

from .config import UploadLimits
from .models import Document, Page


class DocumentError(ValueError):
    """A safe user-facing ingestion failure."""


def safe_filename(name: str) -> str:
    name = PureWindowsPath(str(name).replace("/", "\\")).name
    name = "".join(c for c in name if c.isprintable() and c not in '<>:"|?*')
    name = name.strip(" .")[:180]
    if not name or not name.lower().endswith(".pdf"):
        raise DocumentError("Please choose a file with a .pdf extension.")
    return name


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", " ")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf(data: bytes, filename: str, limits: UploadLimits | None = None) -> Document:
    limits = limits or UploadLimits()
    name = safe_filename(filename)
    if not data:
        raise DocumentError("The uploaded file is empty.")
    if len(data) > limits.max_bytes:
        raise DocumentError(f"PDF exceeds the {limits.max_bytes // (1024 * 1024)} MB limit.")
    if not data.startswith(b"%PDF-"):
        raise DocumentError("File content is not a PDF (missing PDF header).")
    pages, warnings = [], []
    total = 0
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise DocumentError("Password-protected PDFs are not supported. Unlock it first.")
            if not 0 < pdf.page_count <= limits.max_pages:
                raise DocumentError(f"PDF must contain 1..{limits.max_pages} pages.")
            page_count = pdf.page_count
            for number, page in enumerate(pdf, 1):
                text = normalize_text(page.get_text("text", sort=True))
                total += len(text)
                if total > limits.max_characters:
                    raise DocumentError("PDF contains too much extracted text.")
                if text:
                    pages.append(Page(number, text))
                else:
                    warnings.append(f"Page {number} has no extractable text; OCR is not included.")
    except DocumentError:
        raise
    except (RuntimeError, ValueError, pymupdf.FileDataError) as exc:
        raise DocumentError("Cannot read this PDF. It may be damaged or unsupported.") from exc
    if not pages:
        raise DocumentError("No extractable text found. The PDF is empty or scanned; OCR is required.")
    return Document(sha256(data).hexdigest(), name, page_count, tuple(pages), tuple(warnings))
