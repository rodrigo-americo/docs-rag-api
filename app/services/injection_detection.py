import re

# Heurística de baixo recall / baixo falso-positivo, não um classificador
# semântico: cada padrão exige uma frase quase completa (verbo + objeto),
# nunca uma palavra solta, pra não rejeitar juridiquês legítimo que
# mencione "sistema" ou "ignorar" fora desse contexto. Pega tentativas
# óbvias de manipular o LLM via conteúdo de documento; não pretende (nem
# tenta) detectar prompt injection sofisticado ou indireto — ver README.
_SUSPICIOUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # PT-BR — instruções diretas ao modelo
        r"ignor[ae][r]?\s+(todas\s+)?as\s+instru[çc][õo]es\s+(anteriores|acima|do\s+sistema)",
        r"desconsider[ae][r]?\s+as\s+instru[çc][õo]es\s+(anteriores|acima)",
        r"esque[çc]a\s+(tudo|as\s+instru[çc][õo]es)",
        r"voc[êe]\s+(agora\s+)?[ée]\s+um[a]?\s+(assistente|ia|modelo)",
        r"a\s+partir\s+de\s+agora,?\s+(voc[êe]\s+deve|responda|ignore)",
        r"n[ãa]o\s+siga\s+(as\s+)?instru[çc][õo]es\s+(anteriores|do\s+sistema)",
        # EN — cobertura complementar (documento em inglês ou payload
        # traduzido de template conhecido)
        r"ignore\s+(all\s+|the\s+)?(previous|above|prior)\s+instructions",
        r"disregard\s+(the\s+)?(above|previous|prior)",
        r"you\s+are\s+now\s+(a|an)\s+\w+",
        r"new\s+instructions?\s*:",
        r"system\s+prompt\s*:",
        # Tentativa de escapar o delimitador estrutural do prompt
        # (app/rag/graph.py envolve o contexto em <context>...</context>)
        r"</?context>",
    )
]


def find_suspicious_pattern(text: str) -> str | None:
    """Retorna o trecho que casou com o 1º padrão suspeito encontrado, ou None.

    Checagem por chunk (não por documento inteiro) permite apontar qual
    trecho disparou o bloqueio, em vez de só "algo no arquivo é suspeito".
    """
    for pattern in _SUSPICIOUS_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None
