"""RAG answer generation and synthesis with LLM support and an extractive fallback."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from pipeline import (
    _context_for_synthesis,
    synthesize_extractive_answer,
)

try:
    import cloud_ragas
except ImportError:
    cloud_ragas = None  # type: ignore


def _build_prompt(
    query: str,
    contexts: Sequence[str],
    sources: Sequence[dict[str, Any]],
    intent: str | None = None,
    fine_tune_metadata: Any | None = None,
) -> str:
    """Build a rich, document-structured prompt for the LLM.

    Args:
        query: The user's question.
        contexts: List of retrieved context passages.
        sources: List of per-source metadata dicts (source_doc, score, text).
        intent: Optional detected intent (stack, overview, architecture, setup, security, evaluation, fine_tune, general).
        fine_tune_metadata: Structured fine-tuning metadata extracted from HuggingFace model cards.

    Returns:
        The formatted prompt string.
    """
    intent_label = intent or "general"
    fine_tune_rules = (
        "6. If the question is about fine-tuning (dataset, base model, training, LoRA), "
        "extract the exact dataset name, base model, and training details from the Model Card below.\n"
        "7. Answer directly: which dataset was used, who developed it, and the evaluation metrics.\n"
        if intent_label == "fine_tune"
        else ""
    )

    lines: list[str] = [
        "You are an intelligent RAG (Retrieval-Augmented Generation) assistant. "
        "Answer the user's question using ONLY the documents provided below.",
        "",
        "LANGUAGE: Detect the language of the QUESTION and write your ENTIRE answer in that "
        "SAME LANGUAGE. If the question is in Portuguese, answer in Portuguese; if in English, "
        "answer in English; and so on. Never reply in a different language than the question.",
        "",
        "RULES:",
        "1. Do NOT invent information. Use ONLY the content of the documents.",
        "2. Do NOT return raw JSON, code listings, or snippets without explanation.",
        "3. Synthesize the information into a complete, coherent natural-language answer.",
        "4. If the information is insufficient, clearly state that there is not enough evidence.",
        "5. Do not include citations, source lists, or referenced-document footers in the answer body; "
        "the interface will display sources separately.",
        fine_tune_rules,
        "",
        f"QUESTION: {query}",
        "",
    ]

    if intent_label != "general":
        lines.append(f"QUESTION TYPE: {intent_label}")
        lines.append("")

    if fine_tune_metadata is not None:
        summary = fine_tune_metadata.to_summary() if hasattr(fine_tune_metadata, 'to_summary') else str(fine_tune_metadata)
        if summary:
            lines.append("STRUCTURED MODEL METADATA (extracted from the HuggingFace Model Card):")
            lines.append(summary)
            lines.append("")

    lines.append("RETRIEVED DOCUMENTS:")
    lines.append("")

    for idx, context in enumerate(contexts, 1):
        source = sources[idx - 1] if idx - 1 < len(sources) else {}
        source_name = source.get("source_doc", f"document_{idx}")
        cleaned_context = _context_for_synthesis(context)

        lines.append(f"--- Document {idx}: {source_name} ---")
        lines.append(cleaned_context)
        lines.append("")

    lines.append("---")
    lines.append(
        "Based ONLY on the documents above, answer the user's question completely and without "
        "inventing information. Write your entire answer in the SAME LANGUAGE as the QUESTION."
    )
    return "\n".join(lines)


def _llm_available() -> bool:
    return cloud_ragas is not None and bool(cloud_ragas.providers_from_env())


def _get_llm_client() -> Any:
    if not _llm_available():
        raise RuntimeError("LLM is not available. Configure at least one supported cloud provider.")
    return cloud_ragas.FreeTierCloudClient(
        budget=cloud_ragas.CloudCallBudget(max_calls=int(os.getenv("MAX_CLOUD_CHAT_CALLS", "1"))),
        providers=tuple(cloud_ragas.providers_from_env()),
    )


def synthesize_generative_answer(
    query: str,
    contexts: Sequence[str],
    sources: Sequence[dict[str, Any]],
    intent: str | None = None,
    fine_tune_metadata: Any | None = None,
) -> str | None:
    """Synthesize an answer with the LLM using rich context and optional intent detection.

    Args:
        query: The user's question.
        contexts: List of retrieved context passages.
        sources: List of per-source metadata dicts.
        intent: Optional detected intent (stack, overview, architecture, etc).
        fine_tune_metadata: Structured fine-tuning metadata extracted from HuggingFace model cards.

    Returns:
        The synthesized answer, or None if the LLM is unavailable.
    """
    if not _llm_available():
        logger.info("LLM not available (missing supported cloud provider key), skipping generative synthesis")
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


def _fold_for_footer_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _strip_generated_citation_footer(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        folded_line = _fold_for_footer_match(line)
        if re.search(r"^\s*(citacao|citacoes|citation|citations|fontes?|sources?|documentos?\s+referenciados?)\s*:", folded_line):
            folded_footer = _fold_for_footer_match("\n".join(lines[index:]))
            if re.search(r"\bdocumento\s+\d+\b", folded_footer) or re.search(r"\b(readme|package\.json|\.tsx?|\.py|\.md)\b", folded_footer):
                return "\n".join(lines[:index]).rstrip()
    return text


def _post_process_llm_response(raw: str) -> str | None:
    text = _strip_wrapping_code_fences(raw)
    text = _strip_generated_citation_footer(text)
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
    """LLM-first synthesis with an intelligent fallback to extractive.

    Primary path: synthesize_generative_answer (LLM with rich context).
    Fallback: synthesize_extractive_answer (only when the LLM is unavailable).

    Args:
        query: The user's question.
        contexts: List of retrieved context passages.
        sources: List of per-source metadata dicts.
        intent: Optional detected intent.
        fine_tune_metadata: Structured fine-tuning metadata extracted from HuggingFace model cards.

    Returns:
        The synthesized answer (LLM or extractive).
    """
    answer = synthesize_generative_answer(query, contexts, sources, intent, fine_tune_metadata)
    if answer is not None:
        return answer

    # Improved extractive fallback.
    logger.info("LLM unavailable, falling back to extractive synthesis")
    return synthesize_extractive_answer(query, contexts, max_sentences=5)
