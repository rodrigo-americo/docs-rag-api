from pathlib import Path

from app.core.exceptions import UnsupportedFileTypeError

TEXT_EXTENSIONS = (".md", ".markdown", ".txt")


class PlainTextParser:
    """Texto puro (Markdown/TXT) não tem magic bytes — a única forma
    confiável de reconhecer é pela extensão, seguida de uma decodificação
    UTF-8 estrita (rejeita binário disfarçado de .txt em vez de mascarar
    com errors="replace", que produzia texto corrompido silenciosamente).
    """

    def can_parse(self, content: bytes, filename: str) -> bool:
        return Path(filename).suffix.lower() in TEXT_EXTENSIONS

    def parse(self, content: bytes) -> str:
        try:
            return content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            # Sem `filename` aqui de propósito (ver DocumentParser.parse) —
            # quem chama já loga `filename` junto do erro (ver
            # IngestService.process_document), não precisa duplicar aqui.
            raise UnsupportedFileTypeError("conteúdo não é texto UTF-8 válido") from exc
