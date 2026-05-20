"""Geração e síntese de respostas do RAG com suporte a LLM e fallback extraívo."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
import re
from pathlib import Path
from typing import Any, Sequence

from pipeline import (
    _context_for_synthesis,
    _has_enough_evidence,
    synthesize_extractive_answer,
)

try:
    import gemini_ragas
except ImportError:
    gemini_ragas = None  # type: ignore


def _build_prompt(
    query: str,
    contexts: Sequence[str],
    sources: Sequence[dict[str, Any]],
    intent: str | None = None,
    fine_tune_metadata: Any | None = None,
) -> str:
    """Constrói prompt rico para LLM com contexto estruturado por documento.

    Args:
        query: A pergunta do usuário.
        contexts: Lista de textos de contexto recuperados.
        sources: Lista de dicionários com metadata de cada fonte (source_doc, score, text).
        intent: Intent detection opcional (stack, overview, architecture, setup, security, evaluation, fine_tune, general).
        fine_tune_metadata: Metadados estruturados de fine-tuning extraídos de HuggingFace model cards.

    Returns:
        Prompt formatado para o LLM.
    """
    intent_label = intent or "general"
    fine_tune_rules = (
        "6. Se a pergunta é sobre fine-tuning (dataset, base model, treinamento, LoRA), "
        "extraia o nome exato do dataset, modelo base, e detalhes de treinamento do Model Card a seguir.\n"
        "7. Responda diretamente: qual dataset foi usado, quem desenvolveu, métricas de avaliação.\n"
        if intent_label == "fine_tune"
        else ""
    )

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
        fine_tune_rules,
        "",
        f"PERGUNTA: {query}",
        "",
    ]

    if intent_label != "general":
        lines.append(f"TIPO DE PERGUNTA: {intent_label}")
        lines.append("")

    if fine_tune_metadata is not None:
        summary = fine_tune_metadata.to_summary() if hasattr(fine_tune_metadata, 'to_summary') else str(fine_tune_metadata)
        if summary:
            lines.append("METADADOS ESTRUTURADOS DO MODELO (extraídos do HuggingFace Model Card):")
            lines.append(summary)
            lines.append("")

    lines.append("DOCUMENTOS RECUPERADOS:")
    lines.append("")

    for idx, context in enumerate(contexts, 1):
        source = sources[idx - 1] if idx - 1 < len(sources) else {}
        source_name = source.get("source_doc", f"documento_{idx}")
        cleaned_context = _context_for_synthesis(context)

        lines.append(f"--- Documento {idx}: {source_name} ---")
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
    intent: str | None = None,
    fine_tune_metadata: Any | None = None,
) -> str | None:
    """Sintetiza resposta usando LLM com contexto rico e opcional intent detection.

    Args:
        query: A pergunta do usuário.
        contexts: Lista de textos de contexto recuperados.
        sources: Lista de dicionários com metadata de cada fonte.
        intent: Intent detection opcional (stack, overview, architecture, etc).
        fine_tune_metadata: Metadados estruturados de fine-tuning extraídos de HuggingFace model cards.

    Returns:
        Resposta sintetizada ou None se LLM indisponível.
    """
    if not _llm_available():
        logger.info("LLM not available (missing GEMINI_API_KEY), skipping generative synthesis")
        return None

    if not contexts:
        logger.info("No contexts available for generative synthesis")
        return None

    client = _get_llm_client()
    prompt = _build_prompt(query, contexts, sources, intent, fine_tune_metadata)
    logger.info("Calling LLM for generative synthesis (%d contexts, intent=%s, fine_tune=%s)", len(contexts), intent, fine_tune_metadata is not None)

    try:
        raw_answer = client.generate_text(prompt, temperature=0.2)
    except Exception as exc:
        logger.warning("LLM synthesis failed, falling back to extractive: %s", exc)
        return None

    logger.info("Generative synthesis complete (%d chars)", len(raw_answer))
    return _post_process_llm_response(raw_answer)


def _strip_wrapping_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[\w]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _post_process_llm_response(raw: str) -> str:
    text = _strip_wrapping_code_fences(raw)
    if not text:
        return None
    return text


def synthesize_intelligent_answer(
    query: str,
    contexts: Sequence[str],
    sources: Sequence[dict[str, Any]],
    intent: str | None = None,
    fine_tune_metadata: Any | None = None,
) -> str:
    """LLM-first synthesis with intelligent fallback to extractive.

    Primary path: synthesize_generative_answer (LLM com contexto rico)
    Fallback: synthesize_extractive_answer (apenas quando LLM indisponível)

    Args:
        query: A pergunta do usuário.
        contexts: Lista de textos de contexto recuperados.
        sources: Lista de dicionários com metadata de cada fonte.
        intent: Intent detection opcional.
        fine_tune_metadata: Metadados estruturados de fine-tuning extraídos de HuggingFace model cards.

    Returns:
        Resposta sintetizada (LLM ou extractiva).
    """
    answer = synthesize_generative_answer(query, contexts, sources, intent, fine_tune_metadata)
    if answer is not None:
        return answer

    # Fallback extraívo melhorado
    logger.info("LLM unavailable, falling back to extractive synthesis")
    return synthesize_extractive_answer(query, contexts, max_sentences=5)
