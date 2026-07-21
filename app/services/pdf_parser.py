from io import BytesIO

from pypdf import PdfReader

from app.core.exceptions import PdfParseError


def parse_pdf(content: bytes) -> tuple[str, int]:
    """Extrai texto e número de páginas de um PDF.

    Returns:
        (texto_concatenado, num_paginas)

    Raises:
        PdfParseError: se o PDF não puder ser parseado.
    """
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise PdfParseError(f"inválido ({exc})") from exc

    num_pages = len(reader.pages)
    if num_pages == 0:
        raise PdfParseError("vazio (zero páginas)")

    # Concatena texto de todas as páginas com \n\n como separador.
    # Por que \n\n? Porque o RecursiveCharacterTextSplitter usa \n\n
    # como separador prioritário — preserva o boundary de página.
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise PdfParseError("sem texto extraível (provavelmente scan/imagem)")

    return full_text, num_pages
