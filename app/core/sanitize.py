import bleach


def strip_html(text: str) -> str:
    """Remove qualquer tag HTML/JS de texto vindo do LLM ou de documentos.

    Defesa em profundidade: a API hoje só devolve JSON puro, mas `answer`
    e `citations[].snippet` carregam texto derivado de documentos de
    terceiros (upload). Se um frontend futuro renderizar esses campos sem
    escapar (ex: dangerouslySetInnerHTML), essa sanitização garante que a
    API nunca devolve HTML/JS ativo, independente de quem consome.
    """
    return bleach.clean(text, tags=[], strip=True)
