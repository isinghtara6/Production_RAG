import pytest

from app.core.exceptions import RagServiceError
from app.rag.file_extraction import FileExtractionError, UnsupportedFileTypeError, extract_text


def test_txt_extraction():
    text = extract_text("notes.txt", b"Hello world, this is a test.")
    assert text == "Hello world, this is a test."


def test_md_extraction():
    text = extract_text("README.md", "# Title\n\nSome content.".encode("utf-8"))
    assert "Some content." in text


def test_unsupported_extension_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("image.png", b"\x89PNG...")


def test_empty_file_rejected():
    with pytest.raises(FileExtractionError):
        extract_text("empty.txt", b"   ")


def test_invalid_utf8_rejected():
    with pytest.raises(FileExtractionError):
        extract_text("bad.txt", b"\xff\xfe\x00\x01")


def test_all_extraction_errors_are_service_errors():
    # Every raised error must be a RagServiceError subclass so the global
    # exception handler turns it into a structured JSON response, never a
    # raw 500 traceback.
    for exc_cls in (UnsupportedFileTypeError, FileExtractionError):
        assert issubclass(exc_cls, RagServiceError)


@pytest.mark.skipif(
    pytest.importorskip("pypdf", reason="pypdf not installed") is None, reason="pypdf not installed"
)
def test_pdf_extraction_with_real_pdf(tmp_path):
    reportlab = pytest.importorskip("reportlab", reason="reportlab not installed (test-only fixture generator)")
    from reportlab.pdfgen import canvas as rl_canvas

    pdf_path = tmp_path / "test.pdf"
    c = rl_canvas.Canvas(str(pdf_path))
    c.drawString(72, 700, "Cats are small domesticated carnivorous mammals.")
    c.showPage()
    c.save()

    text = extract_text("test.pdf", pdf_path.read_bytes())
    assert "Cats are small" in text


@pytest.mark.skipif(
    pytest.importorskip("docx", reason="python-docx not installed") is None, reason="python-docx not installed"
)
def test_docx_extraction_with_real_docx(tmp_path):
    import docx as python_docx

    docx_path = tmp_path / "test.docx"
    d = python_docx.Document()
    d.add_paragraph("Dogs are domesticated mammals, not natural wild animals.")
    d.save(str(docx_path))

    text = extract_text("test.docx", docx_path.read_bytes())
    assert "Dogs are domesticated" in text
