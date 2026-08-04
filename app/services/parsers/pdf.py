from app.core.config import settings
from app.core.exceptions import DocumentTooManyPagesError
from app.services.pdf_parser import parse_pdf

# Todo PDF válido começa com essa assinatura, independente da extensão
# do nome do arquivo (que o cliente pode errar ou forjar).
PDF_MAGIC_BYTES = b"%PDF-"


class PdfParser:
    def can_parse(self, content: bytes, filename: str) -> bool:
        return content.startswith(PDF_MAGIC_BYTES)

    def parse(self, content: bytes) -> str:
        text, num_pages = parse_pdf(content)
        if num_pages > settings.max_pdf_pages:
            raise DocumentTooManyPagesError(num_pages, settings.max_pdf_pages)
        return text  # pragma: no cover — ver docs/testes.md (mesmo bug de app/api/documents.py)
