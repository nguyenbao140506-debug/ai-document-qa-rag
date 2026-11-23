from dataclasses import replace

import pymupdf
import pytest

from ragqa.config import UploadLimits
from ragqa.pdf_loader import DocumentError, extract_pdf, normalize_text, safe_filename


def test_pdf_extraction_and_metadata(make_pdf):
    data = make_pdf(["Page one contains the battery policy.", "Page two contains a safety rule."])
    doc = extract_pdf(data, "manual.pdf")
    assert doc.filename == "manual.pdf" and doc.page_count == 2
    assert [p.number for p in doc.pages] == [1, 2]
    assert "battery policy" in doc.pages[0].text
    assert doc.id == extract_pdf(data, "renamed.pdf").id


@pytest.mark.parametrize(
    "name,expected",
    [
        ("../../report.pdf", "report.pdf"),
        (r"C:\private\report.pdf", "report.pdf"),
        ("<report>.pdf", "report.pdf"),
        ("Report.PDF", "Report.PDF"),
    ],
)
def test_names_never_become_paths(name, expected):
    assert safe_filename(name) == expected


@pytest.mark.parametrize("name", ["", "doc.exe", "doc.pdf.exe", "..", "a" * 190 + ".pdf"])
def test_bad_extensions_rejected(name):
    with pytest.raises(DocumentError):
        safe_filename(name)


@pytest.mark.parametrize("data", [b"", b"hello", b"MZ binary file", b"%PDF-1.7\ninvalid"])
def test_invalid_pdf_bytes(data):
    with pytest.raises(DocumentError):
        extract_pdf(data, "file.pdf")


def test_byte_and_page_limits(make_pdf):
    data = make_pdf(["one", "two"])
    with pytest.raises(DocumentError, match="MB limit"):
        extract_pdf(data, "a.pdf", replace(UploadLimits(), max_bytes=10))
    with pytest.raises(DocumentError, match="pages"):
        extract_pdf(data, "a.pdf", replace(UploadLimits(), max_pages=1))


def test_text_limit(make_pdf):
    with pytest.raises(DocumentError, match="too much"):
        extract_pdf(make_pdf(["A long test sentence"]), "a.pdf", replace(UploadLimits(), max_characters=3))


def test_empty_pdf(make_pdf):
    with pytest.raises(DocumentError, match="OCR"):
        extract_pdf(make_pdf([""]), "empty.pdf")


def test_image_only_pdf():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 20, 20), False)
        pixmap.clear_with(180)
        page.insert_image((20, 20, 200, 200), pixmap=pixmap)
        data = pdf.tobytes()
    with pytest.raises(DocumentError, match="scanned"):
        extract_pdf(data, "scan.pdf")


def test_partial_blank_pdf_keeps_original_page_numbers(make_pdf):
    doc = extract_pdf(make_pdf(["", "The answer is on page two."]), "partial.pdf")
    assert doc.pages[0].number == 2
    assert doc.page_count == 2 and len(doc.warnings) == 1


def test_encrypted_pdf(make_pdf):
    with pytest.raises(DocumentError, match="Password-protected"):
        extract_pdf(make_pdf(["secret"], encrypted=True), "locked.pdf")


def test_normalization():
    assert normalize_text("A  hy-\nphen\x00 and\t text") == "A hyphen and text"
