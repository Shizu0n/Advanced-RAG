"""Geração e síntese de respostas do RAG com suporte a LLM e fallback extraívo."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

from pipeline import (
    _context_for_synthesis,
    _has_enough_evidence,
    EMPTY_LOCAL_CONTEXT_ANSWER,
    LOW_EVIDENCE_ANSWER,
    synthesize_extractive_answer,
)

try:
    import gemini_ragas
except ImportError:
    gemini_ragas = None  # type: ignore


def _build_prompt(query: str, contexts: Sequence[str], sources: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = [
        "Você é um assistente inteligente de RAG (Retrieval-Augmented Generation). "
        "Sua tarefa é responder à pergunta do usuário usando APENAS os documentos fornecidos abaixo.",
        "",
        "REGRAS:",
        "1. NÃO invente informações. Use SOMENTE o conteúdo dos documentos.",
        "2. NÃO retorne JSON cru, listas de código ou snippets sem explicação.",
        "3. Sintetize as informações em uma resposta completa e coerente em linguagem natural.",
        "4. Se a informação for insuficiente, diga claramente que não há evidências suficientes.",
        "5. Ao final, cite os documentos referenciados.",
        "",
        f"PERGUNTA: {query}",
        "",
        "DOCUMENTOS RECUPERADOS:",
    ]

    for idx, context in enumerate(contexts, 1):
        cleaned_context = _context_for_synthesis(context)
        source_name = (
            sources[idx - 1].get("source_doc", f"documento_{idx}")
            if sources and idx - 1 < len(sources)
            else f"documento_{idx}"
        )
        lines.append(f"--- {source_name} ---")
        lines.append(cleaned_context)
        lines.append("")

    lines.append("---")
    lines.append(
        "Com base APENAS nos documentos acima, responda à pergunta do usuário em português, "
        "de forma completa e sem inventar informações."
    )
    return "\n".join(lines)


def _llm_available() -> bool:
    return gemini_ragas is not None and os.getenv("GEMINI_API_KEY") is not None


def _get_llm_client() -> Any:
    if not _llm_available():
        raise RuntimeError("LLM não está disponível. Configure GEMINI_API_KEY.")
    config = gemini_ragas.config_from_env()
    return gemini_ragas.client_from_config(config)


def synthesize_generative_answer(
    query: str,
    contexts: Sequence[str],
    sources: Sequence[dict[str, Any]],
) -> str | None:
    if not _llm_available():
        return None

    if not contexts:
        return EMPTY_LOCAL_CONTEXT_ANSWER

    if not _has_enough_evidence(query, contexts, sources):
        return LOW_EVIDENCE_ANSWER

    client = _get_llm_client()
    prompt = _build_prompt(query, contexts, sources)

    try:
        raw_answer = client.generate_text(prompt, temperature=0.2)
    except Exception:
        # Em caso de falha no LLM, retorna None para fallback extraívo
        return None

    return _post_process_llm_response(raw_answer)


def _post_process_llm_response(raw: str) -> str:
    text = raw.strip()
    # Remove blocos de código markdown
    if text.startswith("```"):
        text = re.sub(r"^```[\w]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # Remove sujeira comum de LLMs
    text = text.strip()
    if not text:
        return None
    return text


def synthesize_intelligent_answer(
    query: str,
    contexts: Sequence[str],
    sources: Sequence[dict[str, Any]],
) -> str:
    answer = synthesize_generative_answer(query, contexts, sources)
    if answer is not None:
        return answer

    # Fallback extraívo melhorado
    return synthesize_extractive_answer(query, contexts, max_sentences=5)
