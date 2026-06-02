class IngestError(Exception):
    """Erro genérico no ingest."""


class DocumentTooLargeError(IngestError):
    """Arquivo excede limite de tamanho."""

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"Arquivo de {size_bytes} bytes excede limite de {max_bytes} bytes"
        )


class DocumentTooManyPagesError(IngestError):
    """PDF excede limite de páginas."""

    def __init__(self, pages: int, max_pages: int) -> None:
        self.pages = pages
        self.max_pages = max_pages
        super().__init__(
            f"PDF de {pages} páginas excede limite de {max_pages}"
        )


class UnsupportedFileTypeError(IngestError):
    """Tipo de arquivo não suportado."""