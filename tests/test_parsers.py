import pytest

from app.core.exceptions import DocumentTooManyPagesError, PdfParseError, UnsupportedFileTypeError
from app.services.parsers import extract_text


def test_extract_text_reads_txt_by_content_not_extension():
    text = extract_text(b"conteudo qualquer", "arquivo.txt")

    assert text == "conteudo qualquer"


def test_extract_text_treats_pdf_magic_bytes_as_pdf_even_with_txt_extension():
    # Conteúdo começa com os magic bytes %PDF- mas o nome do arquivo diz
    # .txt — deve ser tratado como PDF (conteúdo manda, não a extensão) e
    # falhar no parse por não ser um PDF válido de verdade, não por causa
    # da extensão errada.
    with pytest.raises(PdfParseError):
        extract_text(b"%PDF-1.4 nao e um pdf valido", "arquivo.txt")


def test_extract_text_rejects_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(b"conteudo qualquer", "arquivo.exe")


def test_extract_text_rejects_binary_content_disguised_as_txt():
    binary_content = b"\xff\xfe\x00\x01\x02\x03invalid utf-8 \xc0\xc1"

    with pytest.raises(UnsupportedFileTypeError):
        extract_text(binary_content, "disfarcado.txt")


def test_extract_text_rejects_pdf_with_too_many_pages(monkeypatch):
    monkeypatch.setattr("app.services.parsers.pdf.settings.max_pdf_pages", 1)
    monkeypatch.setattr("app.services.parsers.pdf.parse_pdf", lambda content: ("texto", 5))

    with pytest.raises(DocumentTooManyPagesError):
        extract_text(b"%PDF-1.4 conteudo", "arquivo.pdf")
