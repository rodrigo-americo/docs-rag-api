from typing import Protocol


class DocumentParser(Protocol):
    """Um parser por tipo de arquivo suportado.

    `can_parse` decide se este parser reconhece o conteúdo — pelo
    conteúdo em si (magic bytes) quando possível, não só pela extensão
    do nome, que o cliente pode errar ou forjar. `parse` faz a extração
    de texto de fato, e é quem levanta os erros específicos daquele
    tipo (PDF corrompido, texto não-UTF-8, etc).
    """

    def can_parse(self, content: bytes, filename: str) -> bool:
        """Este parser reconhece esse conteúdo?"""
        ...

    def parse(self, content: bytes) -> str:
        """Extrai o texto do conteúdo. Assume que `can_parse` já retornou True."""
        ...
