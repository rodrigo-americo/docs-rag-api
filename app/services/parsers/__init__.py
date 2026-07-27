from app.core.exceptions import UnsupportedFileTypeError
from app.services.parsers.base import DocumentParser
from app.services.parsers.pdf import PdfParser
from app.services.parsers.plain_text import PlainTextParser

# Ordem importa: PDF precisa ser tentado antes de texto puro. Um PDF é
# reconhecido por magic bytes (bem específico); texto puro só pela
# extensão (mais genérico) — se a ordem fosse invertida e um novo parser
# genérico demais entrasse na lista, ele poderia "roubar" arquivos que
# deveriam cair num parser mais específico.
_PARSERS: list[DocumentParser] = [PdfParser(), PlainTextParser()]


def extract_text(content: bytes, filename: str) -> str:
    """Identifica o tipo do arquivo pelo conteúdo (quando possível) e
    extrai o texto. Adicionar suporte a um novo tipo de arquivo (ex:
    .docx) significa criar um parser novo e registrá-lo em _PARSERS —
    nada aqui nem em IngestService precisa mudar além disso.
    """
    for parser in _PARSERS:
        if parser.can_parse(content, filename):
            return parser.parse(content)

    suffix = filename.rsplit(".", 1)[-1] if "." in filename else filename
    raise UnsupportedFileTypeError(f".{suffix}")
