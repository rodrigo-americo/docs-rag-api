from pathlib import Path
from uuid import UUID

from app.core.config import settings


def _upload_path(document_id: UUID) -> Path:
    return Path(settings.uploads_dir) / f"{document_id}.bin"


def save_upload(document_id: UUID, content: bytes) -> Path:
    """Grava os bytes originais do upload em disco, pelo document_id.

    O worker (processo separado) vai ler esse arquivo de volta — os bytes
    não sobrevivem numa closure entre processos diferentes, só num arquivo
    que os dois conseguem acessar.
    """
    path = _upload_path(document_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def read_upload(document_id: UUID) -> bytes:
    return _upload_path(document_id).read_bytes()


def delete_upload(document_id: UUID) -> None:
    _upload_path(document_id).unlink(missing_ok=True)
