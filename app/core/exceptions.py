from fastapi import HTTPException, status


class IngestError(Exception):
    """Erro genérico no ingest.

    `status_code` é definido por subclasse — o endpoint captura só
    `IngestError` e traduz via `to_http()`, sem precisar de um `except`
    por tipo. Adicionar uma exceção nova não exige tocar em documents.py.
    """

    status_code: int = status.HTTP_400_BAD_REQUEST

    def to_http(self) -> HTTPException:
        return HTTPException(status_code=self.status_code, detail=str(self))


class DocumentTooLargeError(IngestError):
    """Arquivo excede limite de tamanho."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(f"Arquivo de {size_bytes} bytes excede limite de {max_bytes} bytes")


class DocumentTooManyPagesError(IngestError):
    """PDF excede limite de páginas."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE

    def __init__(self, pages: int, max_pages: int) -> None:
        self.pages = pages
        self.max_pages = max_pages
        super().__init__(f"PDF de {pages} páginas excede limite de {max_pages}")


class UnsupportedFileTypeError(IngestError):
    """Tipo de arquivo não suportado."""

    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

    def __init__(self, extension: str) -> None:
        self.extension = extension
        super().__init__(f"Extensão {extension!r} não suportada (use .pdf, .md ou .txt)")


class EmptyDocumentError(IngestError):
    """Arquivo sem conteúdo aproveitável."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(f"Documento {filename!r} sem conteúdo após chunking")


class PdfParseError(IngestError):
    """PDF corrompido, vazio ou sem texto extraível."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Falha ao processar PDF: {reason}")


class EmbeddingProviderError(IngestError):
    """Provider de embeddings (ex: OpenAI) falhou ou está indisponível.

    503, não 500: não é um bug interno, é uma dependência externa fora do
    ar. O cliente pode tentar de novo mais tarde — diferente de um 500,
    que sugere um bug que só será corrigido com deploy.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Serviço de embeddings indisponível: {reason}")


class SuspiciousContentError(IngestError):
    """Documento contém um trecho que casa com um padrão de prompt injection.

    422, não 500: é o documento que está sendo rejeitado por conteúdo, não
    um bug do serviço — mesma família de EmptyDocumentError/PdfParseError.
    A heurística é de baixo recall por design (poucos padrões, bem
    específicos) — existe para bloquear tentativas óbvias, não para
    detectar injection sofisticado. Falsos positivos são possíveis; por
    isso o erro devolve o trecho exato que disparou o bloqueio, pra quem
    escreveu o documento legítimo entender o motivo e poder contestar.
    """

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self, filename: str, matched_text: str) -> None:
        self.filename = filename
        self.matched_text = matched_text
        super().__init__(
            f"Documento {filename!r} rejeitado: trecho suspeito de manipulação "
            f"de instrução encontrado ('{matched_text}'). Se isso for um falso "
            f"positivo, revise o texto ao redor desse trecho e tente novamente."
        )


class DuplicateChunkError(IngestError):
    """Violação da unique constraint (document_id, chunk_index).

    Não é um erro de input do cliente — os índices vêm de `enumerate()`
    sobre uma lista já em memória, então só pode acontecer por retry ou
    corrida em nível de infraestrutura. Por isso 500, não 409: o cliente
    não tem uma ação corretiva óbvia a tomar.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(
            f"Conflito interno ao indexar '{filename}': chunk duplicado para o mesmo documento"
        )
