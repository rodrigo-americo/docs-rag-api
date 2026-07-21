from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.core.exceptions import PdfParseError
from app.services.pdf_parser import parse_pdf


def _make_pdf(num_pages: int = 1, with_text: bool = True) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        if with_text:
            # pypdf não tem API simples de desenhar texto sem reportlab;
            # usamos add_blank_page + injeção manual de texto via annotation
            # não é necessário aqui — testamos o caminho "sem texto" à parte.
            writer.add_blank_page(width=200, height=200)
        else:
            writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_parse_pdf_invalid_bytes_raises_pdf_parse_error():
    with pytest.raises(PdfParseError) as exc_info:
        parse_pdf(b"isto definitivamente nao e um pdf valido")

    assert exc_info.value.status_code == 422


def test_parse_pdf_zero_pages_raises_pdf_parse_error():
    writer = PdfWriter()
    buf = BytesIO()
    writer.write(buf)

    with pytest.raises(PdfParseError, match="vazio"):
        parse_pdf(buf.getvalue())


def test_parse_pdf_blank_page_without_text_raises_pdf_parse_error():
    content = _make_pdf(num_pages=1, with_text=False)

    with pytest.raises(PdfParseError, match="sem texto extraível"):
        parse_pdf(content)
