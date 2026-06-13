"""Golden dataset generation and no-cost evaluation runner."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import pandas as pd
import requests

import cloud_ragas
from pipeline import (
    LocalRAGPipeline,
    load_local_context_nodes,
    _extract_fine_tune_info,
    _is_code_evidence_source,
    _is_readme_source,
)


PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RAGAS_RESULTS_PATH = EVAL_DIR / "ragas_results.csv"
RAGAS_PER_QUESTION_PATH = EVAL_DIR / "ragas_per_question.csv"
CURRENT_SOURCE_PATH = PROJECT_ROOT / "data" / "current_source.json"
STRATEGIES = ["semantic_only", "bm25_only", "hybrid_no_rerank", "hybrid_rerank"]
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
LEXICAL_METRICS = ["lexical_faithfulness", "lexical_answer_relevancy", "lexical_context_recall", "lexical_context_precision"]
LATENCY_METRICS = ["retrieval_ms", "synthesis_ms", "total_ms"]
LATENCY_SUMMARY_METRICS = ["avg_retrieval_ms", "avg_synthesis_ms", "avg_total_ms"]
REQUIRED_GOLDEN_FIELDS = {"question", "ground_truth", "reference_context", "source_doc"}


def load_current_source() -> dict | None:
    """Load data/current_source.json; return None if missing or malformed."""
    if not CURRENT_SOURCE_PATH.exists():
        return None
    try:
        return json.loads(CURRENT_SOURCE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def invalidate_golden_dataset_if_stale() -> bool:
    """Delete golden dataset and RAGAS results if they were generated for a different source.

    Returns True if invalidation happened.
    """
    current = load_current_source()
    if current is None:
        return False

    if not GOLDEN_DATASET_PATH.exists():
        return False

    try:
        golden = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    if not isinstance(golden, list) or not golden:
        return False

    first_item = golden[0]
    if not isinstance(first_item, dict):
        return False

    golden_slug = first_item.get("source_slug")
    if golden_slug == current.get("source_slug"):
        return False

    # Source changed -- invalidate stale eval artifacts.
    for path in [GOLDEN_DATASET_PATH, RAGAS_RESULTS_PATH, RAGAS_PER_QUESTION_PATH]:
        if path.exists():
            path.unlink()
    return True


class QuestionProvider(Protocol):
    name: str

    def generate(self, node: Any) -> dict[str, str]:
        ...


def _node_text(node: Any) -> str:
    if hasattr(node, "get_content"):
        return node.get_content()
    return getattr(node, "text", str(node))


def _source_doc(node: Any) -> str:
    metadata = getattr(node, "metadata", {}) or {}
    return (
        metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("file_path")
        or getattr(node, "node_id", "unknown")
    )


def _display_source_doc(source: str) -> str:
    path = Path(str(source))
    if path.is_absolute():
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.name or str(source).replace("\\", "/")
    return str(source).replace("\\", "/")


def _is_test_question_source(source: str) -> bool:
    source_key = str(source).lower().replace("\\", "/")
    return any(
        marker in source_key
        for marker in (
            ".spec.",
            ".test.",
            "/__tests__/",
            "/tests/",
            "test_",
            "_test.",
        )
    )


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if len(part.strip()) > 20]


QUESTION_STOPWORDS = {
    "about",
    "above",
    "after",
    "before",
    "context",
    "does",
    "explain",
    "from",
    "reference",
    "section",
    "this",
    "what",
    "with",
    "para",
    "pela",
    "pelo",
    "como",
    "esta",
    "este",
    "sobre",
    "contexto",
}

INSTRUCTION_GROUND_TRUTH_PREFIXES = (
    "identify ",
    "summarize ",
    "describe ",
    "use the ",
    "use this ",
    "list the ",
    "explain ",
    "extract ",
    "find ",
    "determine ",
    "identifique ",
    "resuma ",
    "descreva ",
    "use o ",
    "use a ",
    "liste ",
    "explique ",
    "extraia ",
    "encontre ",
    "determine ",
)


def _ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def _topic_keywords(text: str, limit: int = 8) -> list[str]:
    words = re.findall(r"\b[^\W\d_][\w-]{3,}\b", text, flags=re.UNICODE)
    keywords: list[str] = []
    seen: set[str] = set()
    for word in words:
        folded = _ascii_fold(word)
        if folded in QUESTION_STOPWORDS or folded in seen:
            continue
        seen.add(folded)
        keywords.append(word)
        if len(keywords) >= limit:
            break
    return keywords


def _content_terms(text: str) -> set[str]:
    return {
        _ascii_fold(word)
        for word in re.findall(r"\b[^\W\d_][\w-]{2,}\b", text, flags=re.UNICODE)
        if _ascii_fold(word) not in QUESTION_STOPWORDS
    }


def _is_instruction_like_ground_truth(ground_truth: str) -> bool:
    normalized = _ascii_fold(ground_truth).strip()
    return any(normalized.startswith(prefix) for prefix in INSTRUCTION_GROUND_TRUTH_PREFIXES)


def _is_trivial_code_fence_ground_truth(ground_truth: str) -> bool:
    stripped = ground_truth.strip()
    if not stripped.startswith("```"):
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return len(lines) <= 2


def _context_support_score(ground_truth: str, reference_context: str) -> float:
    if not ground_truth.strip() or not reference_context.strip():
        return 0.0
    if _ascii_fold(ground_truth) in _ascii_fold(reference_context):
        return 1.0
    answer_terms = _content_terms(ground_truth)
    context_terms = _content_terms(reference_context)
    if not answer_terms or not context_terms:
        answer_numbers = set(re.findall(r"\d{2,4}", ground_truth))
        context_numbers = set(re.findall(r"\d{2,4}", reference_context))
        return 1.0 if answer_numbers and answer_numbers <= context_numbers else 0.0
    return len(answer_terms & context_terms) / len(answer_terms)


def _is_supported_ground_truth(ground_truth: str, reference_context: str) -> bool:
    if _is_instruction_like_ground_truth(ground_truth):
        return False
    if _is_trivial_code_fence_ground_truth(ground_truth):
        return False
    answer_numbers = set(re.findall(r"\d{2,4}", ground_truth))
    context_numbers = set(re.findall(r"\d{2,4}", reference_context))
    if answer_numbers and answer_numbers <= context_numbers:
        return True
    if len(ground_truth.strip()) <= 12:
        return _ascii_fold(ground_truth) in _ascii_fold(reference_context)
    return _context_support_score(ground_truth, reference_context) >= 0.25


def _validated_golden_item(
    question: str,
    ground_truth: str,
    reference_context: str,
    source_doc: str,
) -> dict[str, str] | None:
    question = question.strip()
    ground_truth = ground_truth.strip()
    reference_context = reference_context.strip()
    if not question or not ground_truth or not reference_context:
        return None
    if not _is_supported_ground_truth(ground_truth, reference_context):
        return None
    return {
        "question": question,
        "ground_truth": ground_truth,
        "reference_context": reference_context,
        "source_doc": source_doc,
    }


@dataclass
class OfflineExtractiveQuestionProvider:
    name: str = "offline_extractive"

    def generate(self, node: Any) -> dict[str, str]:
        text = _node_text(node)
        sentences = _sentences(text)
        answer = max(sentences or [text.strip()], key=len)[:700]
        keywords = _topic_keywords(answer)
        topic = " ".join(keywords[:4]) or "this Python documentation section"
        return {
            "question": f"What does the reference context explain about {topic}?",
            "ground_truth": answer,
        }


@dataclass
class CloudQuestionProvider:
    cloud_client: cloud_ragas.FreeTierCloudClient | None = None
    cache_dir: Path = cloud_ragas.DEFAULT_CACHE_DIR
    max_calls: int = 120
    name: str = "cloud_free_tier"

    @property
    def enabled(self) -> bool:
        return bool(self.cloud_client) or (
            cloud_ragas.cloud_ragas_enabled() and os.getenv("ALLOW_CLOUD_FREE_TIER") == "1"
        )

    def generate(self, node: Any) -> dict[str, str]:
        if not self.enabled:
            raise RuntimeError("Cloud RAGAS disabled; set USE_CLOUD_FREE_TIER_RAGAS=1 and ALLOW_CLOUD_FREE_TIER=1.")

        client = self.cloud_client or cloud_ragas.client_from_config(cloud_ragas.config_from_env())
        context = _node_text(node)[:6000]
        prompt = (
            "Generate one evaluation question and one ground truth answer from this context. "
            "Return strict JSON with keys question and ground_truth.\n\n"
            f"Context:\n{context}"
        )
        data = client.generate_json(prompt)
        return {"question": data["question"], "ground_truth": data["ground_truth"]}

    def refine(self, item: dict[str, str]) -> dict[str, str]:
        if not self.enabled:
            raise RuntimeError("Cloud RAGAS disabled; set USE_CLOUD_FREE_TIER_RAGAS=1 and ALLOW_CLOUD_FREE_TIER=1.")

        client = self.cloud_client or cloud_ragas.client_from_config(cloud_ragas.config_from_env())
        prompt = (
            "Improve this RAG evaluation item if needed. Return strict JSON with keys "
            "question and ground_truth. The ground_truth must be a factual answer, not an "
            "instruction. Use only facts present in the context.\n\n"
            f"Question candidate:\n{item['question']}\n\n"
            f"Ground truth candidate:\n{item['ground_truth']}\n\n"
            f"Context:\n{item['reference_context'][:6000]}"
        )
        data = client.generate_json(prompt)
        return {"question": data["question"], "ground_truth": data["ground_truth"]}


def default_question_providers(
    cloud_client: cloud_ragas.FreeTierCloudClient | None = None,
) -> list[QuestionProvider]:
    if cloud_client is None and cloud_ragas.cloud_ragas_enabled() and os.getenv("ALLOW_CLOUD_FREE_TIER") == "1":
        cloud_client = cloud_ragas.client_from_config(cloud_ragas.config_from_env())
    return [CloudQuestionProvider(cloud_client=cloud_client), OfflineExtractiveQuestionProvider()]


def generate_golden_item(node: Any, providers: Sequence[QuestionProvider]) -> dict[str, str]:
    errors: list[str] = []
    for provider in providers:
        try:
            generated = provider.generate(node)
            question = generated["question"].strip()
            ground_truth = generated["ground_truth"].strip()
            if not question or not ground_truth:
                raise ValueError("provider returned empty question or answer")
            return {
                "question": question,
                "ground_truth": ground_truth,
                "reference_context": _node_text(node),
                "source_doc": _source_doc(node),
                "provider": provider.name,
            }
        except Exception as exc:
            errors.append(f"{provider.name}: {exc}")
    raise RuntimeError("; ".join(errors))


def _cloud_refinement_provider(providers: Sequence[QuestionProvider]) -> CloudQuestionProvider | None:
    for provider in providers:
        if isinstance(provider, CloudQuestionProvider) and provider.enabled:
            return provider
    return None


def _maybe_refine_golden_item_with_cloud(
    item: dict[str, str],
    provider: CloudQuestionProvider | None,
) -> dict[str, str]:
    if provider is None:
        return item
    try:
        refined = provider.refine(item)
    except Exception:
        return item
    validated = _validated_golden_item(
        refined.get("question", ""),
        refined.get("ground_truth", ""),
        item["reference_context"],
        item["source_doc"],
    )
    if not validated:
        return item
    for optional_key in ("source_slug", "provider"):
        if optional_key in item:
            validated[optional_key] = item[optional_key]
    validated["provider"] = provider.name
    return validated


def _specificity_score(item: dict[str, str]) -> tuple[int, int]:
    question = item["question"]
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", question.lower())
    return (len(question), len(set(terms)))


def _question_source_group(node: Any) -> str:
    source = _source_doc(node)
    source_key = source.lower().replace("\\", "/")
    if _is_test_question_source(source_key):
        return "other"
    if source_key.endswith(("package.json", "requirements.txt", "pyproject.toml", "cargo.toml", "go.mod")):
        return "manifest"
    if _is_code_evidence_source(source):
        return "code"
    if _is_readme_source(source) or Path(source_key).name == "readme":
        return "readme"
    if Path(source_key).suffix.lower() in {".md", ".rst", ".txt"}:
        return "docs"
    if Path(source_key).suffix.lower() in {".pdf", ".docx", ".doc"}:
        return "document"
    return "other"


def _select_question_source_nodes(nodes: Sequence[Any], limit: int) -> list[Any]:
    if limit <= 0:
        return []

    selectable_nodes = [node for node in nodes if not _is_test_question_source(_source_doc(node))] or list(nodes)
    grouped: dict[str, list[Any]] = {"code": [], "docs": [], "manifest": [], "readme": [], "document": [], "other": []}
    for node in selectable_nodes:
        grouped.setdefault(_question_source_group(node), []).append(node)

    selected: list[Any] = []
    seen: set[str] = set()
    for group in ("code", "docs", "manifest", "readme", "document", "other"):
        for node in grouped.get(group, []):
            key = f"{_source_doc(node)}::{getattr(node, 'node_id', id(node))}"
            if key in seen:
                continue
            selected.append(node)
            seen.add(key)
            break

    for node in selectable_nodes:
        if len(selected) >= limit:
            break
        key = f"{_source_doc(node)}::{getattr(node, 'node_id', id(node))}"
        if key in seen:
            continue
        selected.append(node)
        seen.add(key)

    return selected[:limit]


@dataclass(frozen=True)
class CompositeQuestionNode:
    text: str
    node_id: str
    metadata: dict[str, str]

    def get_content(self) -> str:
        return self.text


def _build_composite_question_nodes(nodes: Sequence[Any]) -> list[Any]:
    grouped: dict[str, list[Any]] = {"readme": [], "manifest": [], "code": [], "docs": [], "document": [], "other": []}
    for node in nodes:
        grouped.setdefault(_question_source_group(node), []).append(node)

    readme = grouped.get("readme", [])[:1]
    docs = grouped.get("docs", [])[:1]
    manifests = grouped.get("manifest", [])[:2]
    code = grouped.get("code", [])[:2]
    composites: list[Any] = []
    if readme and (docs or manifests or code):
        parts = [*readme, *docs, *manifests, *code]
        context = "\n\n".join(
            f"Source: {_source_doc(node)}\n{_node_text(node)[:1200]}"
            for node in parts
        ).strip()
        composites.append(
            CompositeQuestionNode(
                text=context,
                node_id="composite-project-readme-and-source",
                metadata={"file_name": "composite:README+source-evidence"},
            )
        )
    return composites


def _combined_question_context(nodes: Sequence[Any]) -> str:
    return "\n\n".join(_node_text(node)[:1000] for node in list(nodes)[:3]).strip()


def _context_lines(text: str) -> list[str]:
    heading_patterns = (
        "Perfil Profissional",
        "Professional Profile",
        "Formação Acadêmica",
        "Formacao Academica",
        "Education",
        "Projetos e Experiência Prática",
        "Projetos e Experiencia Pratica",
        "Projects",
        "Habilidades Técnicas",
        "Habilidades Tecnicas",
        "Technical Skills",
        "Skills",
        "Idiomas",
        "Languages",
        "Certificações",
        "Certificacoes",
        "Certifications",
    )
    prepared = text.replace("\r\n", "\n").replace("\r", "\n")
    prepared = re.sub(r"\s*[•]\s*", "\n•", prepared)
    for heading in heading_patterns:
        prepared = re.sub(rf"(?<!\n)({re.escape(heading)})", r"\n\1", prepared, flags=re.IGNORECASE)
    lines: list[str] = []
    for raw_line in prepared.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip(" \t-*•")).strip()
        if line:
            lines.append(line)
    if lines:
        return lines
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _line_has_any(line: str, markers: Sequence[str]) -> bool:
    folded = _ascii_fold(line)
    return any(marker in folded for marker in markers)


def _join_answer_lines(lines: Sequence[str], limit: int = 900) -> str:
    answer = " ".join(line.strip() for line in lines if line.strip())
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer[:limit].rstrip(" ,;")


def _section_lines(
    context: str,
    start_markers: Sequence[str],
    stop_markers: Sequence[str],
    limit: int = 8,
) -> list[str]:
    lines = _context_lines(context)
    selected: list[str] = []
    collecting = False
    for line in lines:
        if not collecting and _line_has_any(line, start_markers):
            collecting = True
            if ":" in line:
                remainder = line.split(":", 1)[1].strip()
                if remainder:
                    selected.append(remainder)
            continue
        if not collecting:
            continue
        if selected and _line_has_any(line, stop_markers):
            break
        selected.append(line)
        if len(selected) >= limit:
            break
    if selected:
        return selected

    folded = _ascii_fold(context)
    starts = [
        (folded.find(marker), marker)
        for marker in start_markers
        if folded.find(marker) >= 0
    ]
    if not starts:
        return []
    start_index, marker = min(starts, key=lambda item: item[0])
    content_start = start_index + len(marker)
    stop_indexes = [
        folded.find(stop_marker, content_start)
        for stop_marker in stop_markers
        if folded.find(stop_marker, content_start) >= 0
    ]
    content_end = min(stop_indexes) if stop_indexes else len(context)
    snippet = context[content_start:content_end]
    snippet_lines = [
        re.sub(r"\s+", " ", part.strip(" \t-*•")).strip()
        for part in re.split(r"\n+|(?<=[.!?])\s+", snippet)
        if part.strip(" \t-*•")
    ]
    return snippet_lines[:limit]


def _lines_with_markers(context: str, markers: Sequence[str], limit: int = 8) -> list[str]:
    selected = [line for line in _context_lines(context) if _line_has_any(line, markers)]
    return selected[:limit]


def _colon_fact_lines(context: str, markers: Sequence[str] | None = None, limit: int = 8) -> list[str]:
    selected: list[str] = []
    for line in _context_lines(context):
        if ":" not in line:
            continue
        if markers and not _line_has_any(line, markers):
            continue
        key, value = line.split(":", 1)
        if len(value.strip()) < 2:
            continue
        selected.append(f"{key.strip()}: {value.strip()}")
        if len(selected) >= limit:
            break
    return selected


def _salient_sentences(context: str, markers: Sequence[str] | None = None, limit: int = 3) -> list[str]:
    sentences = _sentences(context) or _context_lines(context)
    selected: list[str] = []
    for sentence in sentences:
        if markers and not _line_has_any(sentence, markers):
            continue
        selected.append(sentence)
        if len(selected) >= limit:
            break
    if selected:
        return selected
    return sentences[:limit]


def _answer_from_section(
    context: str,
    start_markers: Sequence[str],
    stop_markers: Sequence[str],
    fallback_markers: Sequence[str] | None = None,
    limit: int = 8,
) -> str:
    lines = _section_lines(context, start_markers, stop_markers, limit=limit)
    if not lines and fallback_markers:
        lines = _colon_fact_lines(context, fallback_markers, limit=limit)
    if not lines and fallback_markers:
        lines = _lines_with_markers(context, fallback_markers, limit=limit)
    return _join_answer_lines(lines)


RESUME_STOP_MARKERS = (
    "perfil profissional",
    "professional profile",
    "formacao academica",
    "educacao",
    "education",
    "projetos",
    "experiencia pratica",
    "experience",
    "habilidades tecnicas",
    "technical skills",
    "skills",
    "idiomas",
    "languages",
    "certificacoes",
    "certifications",
)


def _resume_skills_answer(context: str) -> str:
    return _answer_from_section(
        context,
        ("habilidades tecnicas", "technical skills", "skills"),
        ("idiomas", "languages", "educacao", "education", "certificacoes", "certifications"),
        fallback_markers=("linguagens", "frontend", "backend", "banco", "devops", "metodologias", "languages", "database", "tools"),
    )


def _resume_experience_answer(context: str) -> str:
    return _answer_from_section(
        context,
        ("perfil profissional", "professional profile", "experiencia profissional", "professional experience"),
        ("formacao academica", "education", "projetos", "projects", "habilidades"),
        fallback_markers=("experiencia", "experience", "desenvolvedor", "developer", "apis", "arquitetura"),
        limit=4,
    )


def _resume_projects_answer(context: str) -> str:
    return _answer_from_section(
        context,
        ("projetos", "experiencia pratica", "projects", "practical experience"),
        ("habilidades tecnicas", "technical skills", "skills", "idiomas", "languages"),
        fallback_markers=("github.com", "system", "sistema", "project", "projeto"),
        limit=10,
    )


def _resume_education_answer(context: str) -> str:
    return _answer_from_section(
        context,
        ("formacao academica", "educacao", "education"),
        ("projetos", "experience", "habilidades", "skills"),
        fallback_markers=("bacharelado", "universidade", "university", "graduation", "2027", "certificacao", "certification"),
        limit=5,
    )


def _document_topics_answer(context: str) -> str:
    heading_like = [
        line
        for line in _context_lines(context)
        if len(line.split()) <= 8 and not line.endswith(".") and not line.endswith(",")
    ][:5]
    return _join_answer_lines(heading_like or _salient_sentences(context, limit=3))


def _document_key_facts_answer(context: str) -> str:
    return _join_answer_lines(_colon_fact_lines(context, limit=6) or _salient_sentences(context, limit=3))


def _document_responsibilities_answer(context: str) -> str:
    markers = (
        "implementou",
        "desenvolveu",
        "projetou",
        "estruturou",
        "responsavel",
        "responsibility",
        "implemented",
        "developed",
        "designed",
        "built",
        "outcome",
        "decision",
    )
    return _join_answer_lines(_lines_with_markers(context, markers, limit=5) or _salient_sentences(context, limit=2))


def _project_stack_answer(context: str) -> str:
    markers = (
        "stack",
        "tools",
        "dependencies",
        "devdependencies",
        "framework",
        "frontend",
        "backend",
        "python",
        "streamlit",
        "chromadb",
        "react",
        "node",
        "typescript",
        "java",
        "spring",
    )
    return _join_answer_lines(_colon_fact_lines(context, markers, limit=8) or _lines_with_markers(context, markers, limit=8))


def _project_setup_answer(context: str) -> str:
    markers = ("setup", "install", "run", "pip ", "python ", "streamlit", "npm", "docker", "requirements")
    return _join_answer_lines(_lines_with_markers(context, markers, limit=8))


def _project_architecture_answer(context: str) -> str:
    markers = ("architecture", "arquitetura", "layer", "camada", "module", "component", "service", "repository")
    return _join_answer_lines(_lines_with_markers(context, markers, limit=8))


def _project_troubleshooting_answer(context: str) -> str:
    markers = ("troubleshoot", "error", "erro", "runtime", "setup", "install", "failed", "falha", "warning")
    return _join_answer_lines(_lines_with_markers(context, markers, limit=6))


def _source_set_is_project_like(nodes: Sequence[Any]) -> bool:
    groups = {_question_source_group(node) for node in nodes}
    return bool(groups & {"code", "manifest", "readme"})


def _source_set_is_resume_like(nodes: Sequence[Any]) -> bool:
    source_text = " ".join(_source_doc(node) for node in nodes)
    body_text = " ".join(_node_text(node)[:1500] for node in list(nodes)[:4])
    folded = _ascii_fold(f"{source_text} {body_text}")
    resume_markers = {
        "curriculo",
        "resume",
        "curriculum",
        "experiencia",
        "habilidades",
        "formacao",
        "educacao",
        "certificacoes",
        "professional experience",
        "technical skills",
    }
    return any(marker in folded for marker in resume_markers)


def _source_set_is_model_card_like(nodes: Sequence[Any]) -> bool:
    source_text = " ".join(_source_doc(node) for node in nodes)
    body_text = "\n".join(_node_text(node)[:3000] for node in list(nodes)[:4])
    folded = _ascii_fold(f"{source_text}\n{body_text}")
    has_model_metadata = "base_model:" in folded or "model card" in folded or "hugging face" in folded
    has_training_signal = any(marker in folded for marker in ("datasets:", "dataset:", "fine-tuned", "fine tuned", "lora", "qlora", "exact match"))
    return has_model_metadata and has_training_signal


def _fine_tune_metadata_for_nodes(nodes: Sequence[Any]):
    contexts = [_node_text(node) for node in nodes]
    sources = [{"source_doc": _source_doc(node)} for node in nodes]
    return _extract_fine_tune_info(contexts, sources)


def _model_card_reference_context(nodes: Sequence[Any]) -> str:
    return "\n\n".join(_node_text(node)[:2500] for node in list(nodes)[:4]).strip()


def _fine_tune_dataset_text(dataset: Any) -> str:
    if isinstance(dataset, list):
        return ", ".join(str(item) for item in dataset if str(item).strip())
    return str(dataset).strip() if dataset else ""


def _fine_tune_mapping_text(values: dict[str, Any], limit: int = 6) -> str:
    parts: list[str] = []
    for key, value in values.items():
        if not str(key).strip() or str(key).lower() == "parameter":
            continue
        if isinstance(value, str) and set(value.strip()) <= {"-", ":"}:
            continue
        parts.append(f"{key}={value}")
        if len(parts) >= limit:
            break
    return ", ".join(parts)


def _model_card_query_items(nodes: Sequence[Any]) -> list[dict[str, str]]:
    context = _model_card_reference_context(nodes)
    source_doc = _source_doc(nodes[0]) if nodes else "unknown"
    metadata = _fine_tune_metadata_for_nodes(nodes)
    candidates: list[tuple[str, str]] = []

    dataset_text = _fine_tune_dataset_text(metadata.dataset)
    if dataset_text:
        candidates.append(
            (
                "What dataset was used to fine-tune this Hugging Face model?",
                f"The fine-tuning dataset is {dataset_text}.",
            )
        )
    if metadata.base_model:
        candidates.append(
            (
                "What base model does this adapter fine-tune?",
                f"The adapter fine-tunes {metadata.base_model}.",
            )
        )
    if metadata.training_details:
        training_text = _fine_tune_mapping_text(metadata.training_details)
        if training_text:
            candidates.append(
                (
                    "What training setup is documented for this fine-tuned model?",
                    f"Training details include {training_text}.",
                )
            )
    if metadata.evaluation_metrics:
        metrics_text = _fine_tune_mapping_text(metadata.evaluation_metrics)
        if metrics_text:
            candidates.append(
                (
                    "What evaluation result is reported for the fine-tuned adapter?",
                    f"The model card reports {metrics_text}.",
                )
            )
    if metadata.lora_config:
        lora_text = _fine_tune_mapping_text(metadata.lora_config)
        if lora_text:
            candidates.append(
                (
                    "What LoRA or QLoRA configuration is documented for this model?",
                    f"The LoRA configuration includes {lora_text}.",
                )
            )

    return [
        item
        for question, answer in candidates
        if (item := _validated_golden_item(question, answer, context, source_doc))
    ]


def _model_card_template_item_for_node(node: Any) -> dict[str, str] | None:
    items = _model_card_query_items([node])
    if not items:
        return None
    item = dict(items[0])
    item["question"] = f"What fine-tuning metadata is documented in {_display_source_doc(_source_doc(node))}?"
    return item


def _resume_query_items(nodes: Sequence[Any]) -> list[dict[str, str]]:
    context = _combined_question_context(nodes)
    source_doc = _source_doc(nodes[0]) if nodes else "unknown"
    candidates = [
        (
            "What technical skills are listed in this resume?",
            _resume_skills_answer(context),
        ),
        (
            "What professional experience is described in this resume?",
            _resume_experience_answer(context),
        ),
        (
            "What projects or achievements are highlighted in this resume?",
            _resume_projects_answer(context),
        ),
        (
            "What education or certifications are included in this resume?",
            _resume_education_answer(context),
        ),
    ]
    return [
        item
        for question, answer in candidates
        if (item := _validated_golden_item(question, answer, context, source_doc))
    ]


def _document_query_items(nodes: Sequence[Any]) -> list[dict[str, str]]:
    context = _combined_question_context(nodes)
    source_doc = _source_doc(nodes[0]) if nodes else "unknown"
    candidates = [
        ("What are the main topics covered in this document?", _document_topics_answer(context)),
        ("What key facts or entities are described in this document?", _document_key_facts_answer(context)),
        (
            "What responsibilities, decisions, or outcomes are mentioned in this document?",
            _document_responsibilities_answer(context),
        ),
        ("What details would a reader need to understand this document?", _join_answer_lines(_salient_sentences(context, limit=3))),
    ]
    return [
        item
        for question, answer in candidates
        if (item := _validated_golden_item(question, answer, context, source_doc))
    ]


def _realistic_user_query_items(nodes: Sequence[Any]) -> list[dict[str, str]]:
    if not nodes:
        return []
    if _source_set_is_model_card_like(nodes):
        return _model_card_query_items(nodes)
    if not _source_set_is_project_like(nodes):
        return _resume_query_items(nodes) if _source_set_is_resume_like(nodes) else _document_query_items(nodes)

    context = _combined_question_context(nodes)
    source_doc = _source_doc(nodes[0]) if nodes else "unknown"
    candidates = [
        ("What stack and tools does this project use?", _project_stack_answer(context)),
        ("How do I set up and run this project?", _project_setup_answer(context)),
        ("What is the architecture of this project?", _project_architecture_answer(context)),
        ("How should I troubleshoot common setup or runtime problems?", _project_troubleshooting_answer(context)),
    ]
    return [
        item
        for question, answer in candidates
        if (item := _validated_golden_item(question, answer, context, source_doc))
    ]


NOISY_QUESTION_TERMS = {
    "modulefileextensions",
    "rootdir",
    "testregex",
    "schemastore",
    "schema url",
    "schema https",
    "json fields",
    "what fields exist",
}


def _template_golden_item_for_node(node: Any) -> dict[str, str] | None:
    source_doc = _source_doc(node)
    display_source = _display_source_doc(source_doc)
    context = _node_text(node).strip()
    if not context:
        return None
    if _source_set_is_model_card_like([node]):
        return _model_card_template_item_for_node(node)
    group = _question_source_group(node)
    if group == "manifest":
        question = f"What dependencies and tools does {display_source} declare for this project?"
        ground_truth = _project_stack_answer(context) or _join_answer_lines(_context_lines(context)[:6])
    elif group == "code":
        question = f"What feature or runtime flow is implemented in {display_source}?"
        ground_truth = _join_answer_lines(_salient_sentences(context, ("class", "def ", "function", "return", "implements", "exposes"), limit=3))
    elif group in {"readme", "docs"}:
        question = f"What setup, architecture, or usage guidance is documented in {display_source}?"
        ground_truth = (
            _project_setup_answer(context)
            or _project_architecture_answer(context)
            or _join_answer_lines(_salient_sentences(context, limit=3))
        )
    else:
        return None
    return _validated_golden_item(question, ground_truth, context[:3000], source_doc)


def _is_noisy_golden_question(item: dict[str, str]) -> bool:
    question = item.get("question", "")
    normalized = question.lower()
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    if any(term.replace(" ", "") in compact for term in NOISY_QUESTION_TERMS):
        return True
    if "reference context" in normalized and any(term in normalized for term in (" import ", " schema ", " transform")):
        return True
    identifiers = re.findall(r"\b[A-Za-z]+(?:[A-Z][A-Za-z0-9]+|_[A-Za-z0-9_]+)\b", question)
    return len(identifiers) >= 2 and not any(term in normalized for term in ("dependency", "feature", "runtime", "architecture"))


def _is_high_quality_golden_item(item: dict[str, str]) -> bool:
    return (
        bool(item.get("question", "").strip())
        and not _is_noisy_golden_question(item)
        and _is_supported_ground_truth(item.get("ground_truth", ""), item.get("reference_context", ""))
    )


def _normalized_question_key(item: dict[str, str]) -> str:
    return re.sub(r"\s+", " ", item.get("question", "").strip().lower())


def filter_best_golden_items(candidates: Sequence[dict[str, str]], limit: int = 30) -> list[dict[str, str]]:
    ranked = sorted((item for item in candidates if _is_high_quality_golden_item(item)), key=_specificity_score, reverse=True)
    filtered: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for item in ranked:
        if len(filtered) >= limit:
            break
        question_key = _normalized_question_key(item)
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        entry: dict[str, str] = {
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "reference_context": item["reference_context"],
            "source_doc": item["source_doc"],
        }
        if "source_slug" in item:
            entry["source_slug"] = item["source_slug"]
        filtered.append(entry)
    return filtered


def generate_golden_dataset(
    nodes: Sequence[Any],
    output_path: Path = GOLDEN_DATASET_PATH,
    providers: Sequence[QuestionProvider] | None = None,
    chunk_limit: int = 50,
    final_limit: int = 30,
) -> list[dict[str, str]]:
    providers = list(providers or default_question_providers())
    current_source = load_current_source()
    source_slug = current_source.get("source_slug", "") if current_source else ""
    question_nodes = _select_question_source_nodes(nodes, chunk_limit)
    composite_nodes = _build_composite_question_nodes(question_nodes or nodes)
    cloud_refinement_provider = _cloud_refinement_provider(providers)
    model_card_source = _source_set_is_model_card_like(question_nodes or nodes)

    candidates = _realistic_user_query_items([*composite_nodes, *(question_nodes or nodes)])
    for node in [*composite_nodes, *question_nodes]:
        if model_card_source:
            template_item = _model_card_template_item_for_node(node)
            if template_item:
                candidates.append(template_item)
            continue
        template_item = _template_golden_item_for_node(node)
        if template_item:
            candidates.append(template_item)
        item = generate_golden_item(node, providers)
        candidates.append(item)
    candidates = [
        _maybe_refine_golden_item_with_cloud(item, cloud_refinement_provider)
        for item in candidates
    ]
    if source_slug:
        for item in candidates:
            item["source_slug"] = source_slug

    realistic_target = 4 if final_limit >= 6 else min(4, max(1, final_limit // 2))
    realistic = [item for item in candidates[:4] if _is_high_quality_golden_item(item)][:realistic_target]
    remaining = filter_best_golden_items(candidates[4:], limit=max(final_limit - len(realistic), 0))
    dataset: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for item in realistic + remaining:
        question_key = _normalized_question_key(item)
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        dataset.append(item)
        if len(dataset) >= final_limit:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset


OPTIONAL_GOLDEN_FIELDS = {"source_slug"}


def validate_golden_dataset(dataset: Any) -> list[dict[str, str]]:
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Golden dataset must be a non-empty list of evaluation records.")

    validated: list[dict[str, str]] = []
    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Golden dataset record {index} must be an object.")

        missing = sorted(REQUIRED_GOLDEN_FIELDS - set(item))
        if missing:
            raise ValueError(f"Golden dataset record {index} is missing required fields: {', '.join(missing)}.")

        cleaned: dict[str, str] = {}
        for field in sorted(REQUIRED_GOLDEN_FIELDS):
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Golden dataset record {index} field {field} must be a non-empty string.")
            cleaned[field] = value.strip()
        for field in OPTIONAL_GOLDEN_FIELDS:
            if field in item and isinstance(item[field], str) and item[field].strip():
                cleaned[field] = item[field].strip()
        validated.append(cleaned)

    return validated


def load_golden_dataset_metadata(path: Path = GOLDEN_DATASET_PATH) -> dict[str, str] | None:
    """Return source_slug and generation date from golden dataset without loading all items.

    Returns None if dataset is missing, empty, or lacks source_slug.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    source_slug = first.get("source_slug")
    if not source_slug:
        return None
    mtime = path.stat().st_mtime
    generation_date = pd.Timestamp(mtime, unit="s").strftime("%Y-%m-%d %H:%M")
    return {"source_slug": source_slug, "generation_date": generation_date, "question_count": str(len(data))}


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict[str, str]]:
    return validate_golden_dataset(json.loads(path.read_text(encoding="utf-8")))


def _terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _overlap_score(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def offline_metric_scores(question: str, answer: str, contexts: Sequence[str], ground_truth: str) -> dict[str, float]:
    joined_context = " ".join(contexts)
    return {
        "lexical_faithfulness": _overlap_score(answer, joined_context),
        "lexical_answer_relevancy": _overlap_score(question, answer),
        "lexical_context_recall": _overlap_score(ground_truth, joined_context),
        "lexical_context_precision": sum(_overlap_score(context, ground_truth) for context in contexts) / max(len(contexts), 1),
    }


def _real_ragas_enabled() -> bool:
    return cloud_ragas.cloud_ragas_enabled()


def _index_build_enabled() -> bool:
    return os.getenv("ALLOW_INDEX_BUILD") == "1"


def evaluate_strategy(
    dataset: Sequence[dict[str, str]],
    strategy: str,
    pipeline: LocalRAGPipeline,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for item in dataset:
        started = time.perf_counter()
        result = pipeline.chat_query(item["question"], strategy=strategy)
        measured_total_ms = (time.perf_counter() - started) * 1000
        scores = offline_metric_scores(
            item["question"],
            result["answer"],
            result["contexts"],
            item["ground_truth"],
        )
        rows.append(
            {
                "strategy": strategy,
                "question": item["question"],
                "answer": result["answer"],
                "ground_truth": item["ground_truth"],
                "contexts": json.dumps(result["contexts"], ensure_ascii=False),
                "source_doc": item["source_doc"],
                "evaluation_backend": "offline_heuristic",
                **_latency_values(result, measured_total_ms),
                **scores,
            }
        )

    summary = {
        metric: sum(row[metric] for row in rows) / max(len(rows), 1)
        for metric in LEXICAL_METRICS
    }
    return summary, rows


def _strict_cloud_ragas_enabled() -> bool:
    return os.getenv("CLOUD_RAGAS_STRICT") == "1"


def _max_real_ragas_rows() -> int:
    raw_value = os.getenv("MAX_REAL_RAGAS_ROWS", "3")
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 3


def maybe_run_real_ragas(
    rows: Sequence[dict[str, Any]],
    cloud_client: cloud_ragas.FreeTierCloudClient | None = None,
    enabled: bool | None = None,
) -> dict[str, float] | None:
    if enabled is False or not _real_ragas_enabled():
        return None

    config = cloud_ragas.config_from_env() if cloud_client is None else None
    client = cloud_client or cloud_ragas.client_from_config(config)
    try:
        budget = getattr(client, "budget", None)
        sampled_rows = list(rows)[: _max_real_ragas_rows()]
        return cloud_ragas.run_ragas(
            sampled_rows,
            cache_dir=config.cache_dir if config else getattr(client, "cache_dir", cloud_ragas.DEFAULT_CACHE_DIR),
            max_calls=config.max_calls if config else getattr(budget, "max_calls", 120),
            cloud_client=client,
        )
    except (
        cloud_ragas.CloudProviderUnavailable,
        requests.exceptions.RequestException,
    ):
        if _strict_cloud_ragas_enabled():
            raise
        return None
    except RuntimeError as exc:
        if _strict_cloud_ragas_enabled():
            raise
        message = str(exc)
        if "MAX_CLOUD_CALLS" in message or "cloud providers unavailable" in message or "Gemini embedding models unavailable" in message:
            return None
        raise


def maybe_run_real_ragas_with_status(
    rows: Sequence[dict[str, Any]],
    cloud_client: cloud_ragas.FreeTierCloudClient | None = None,
    enabled: bool | None = None,
) -> tuple[dict[str, float] | None, str]:
    if enabled is False or not _real_ragas_enabled():
        return None, "Cloud RAGAS was not enabled for this evaluation run."

    config = cloud_ragas.config_from_env() if cloud_client is None else None
    client = cloud_client or cloud_ragas.client_from_config(config)
    try:
        budget = getattr(client, "budget", None)
        sampled_rows = list(rows)[: _max_real_ragas_rows()]
        scores = cloud_ragas.run_ragas(
            sampled_rows,
            cache_dir=config.cache_dir if config else getattr(client, "cache_dir", cloud_ragas.DEFAULT_CACHE_DIR),
            max_calls=config.max_calls if config else getattr(budget, "max_calls", 120),
            cloud_client=client,
        )
        if not scores:
            return None, "Cloud RAGAS returned no scores, so offline metrics were retained."
        return scores, ""
    except (
        cloud_ragas.CloudProviderUnavailable,
        requests.exceptions.RequestException,
    ) as exc:
        if _strict_cloud_ragas_enabled():
            raise
        return None, f"{type(exc).__name__}: {exc}"
    except RuntimeError as exc:
        if _strict_cloud_ragas_enabled():
            raise
        message = str(exc)
        if "MAX_CLOUD_CALLS" in message or "cloud providers unavailable" in message or "Gemini embedding models unavailable" in message:
            return None, message
        raise


def _finite_ragas_metrics(scores: dict[str, float]) -> set[str]:
    finite: set[str] = set()
    for metric in RAGAS_METRICS:
        try:
            value = float(scores[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite.add(metric)
    return finite


def _cloud_ragas_score_status(scores: dict[str, float]) -> tuple[str, str]:
    finite_metrics = _finite_ragas_metrics(scores)
    missing = [metric for metric in RAGAS_METRICS if metric not in finite_metrics]
    if missing:
        return (
            "degraded",
            "Cloud RAGAS returned partial metrics; missing or non-finite: " + ", ".join(missing),
        )
    return "succeeded", ""


def _finite_latency_ms(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(number, 0.0)


def _latency_values(result: dict[str, Any], measured_total_ms: float) -> dict[str, float]:
    trace = result.get("trace") if isinstance(result, dict) else None
    latency = trace.get("latency") if isinstance(trace, dict) else None
    latency = latency if isinstance(latency, dict) else {}

    measured_total_ms = max(float(measured_total_ms), 0.0)
    retrieval_ms = _finite_latency_ms(latency.get("retrieval_ms"))
    synthesis_ms = _finite_latency_ms(latency.get("synthesis_ms"))
    total_ms = _finite_latency_ms(latency.get("total_ms")) or measured_total_ms

    if retrieval_ms is None and synthesis_ms is None:
        retrieval_ms = total_ms
        synthesis_ms = 0.0
    elif retrieval_ms is None:
        retrieval_ms = max(total_ms - float(synthesis_ms), 0.0)
    elif synthesis_ms is None:
        synthesis_ms = max(total_ms - float(retrieval_ms), 0.0)

    return {
        "retrieval_ms": round(float(retrieval_ms), 3),
        "synthesis_ms": round(float(synthesis_ms), 3),
        "total_ms": round(max(total_ms, float(retrieval_ms) + float(synthesis_ms)), 3),
    }


def _latency_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for detail_metric, summary_metric in zip(LATENCY_METRICS, LATENCY_SUMMARY_METRICS, strict=True):
        values = [
            float(row[detail_metric])
            for row in rows
            if detail_metric in row and math.isfinite(float(row[detail_metric]))
        ]
        summary[summary_metric] = round(sum(values) / max(len(values), 1), 3)
    return summary


def run_evaluation(
    golden_path: Path = GOLDEN_DATASET_PATH,
    pipeline: LocalRAGPipeline | None = None,
    cloud_client: cloud_ragas.FreeTierCloudClient | None = None,
    use_real_ragas: bool | None = None,
    progress_callback=None,
) -> dict[str, dict[str, float]]:
    logger.info("run_evaluation: loading golden dataset from %s", golden_path)
    dataset = load_golden_dataset(golden_path)
    logger.info("run_evaluation: %d golden items, %d strategies", len(dataset), len(STRATEGIES))
    pipeline = pipeline or LocalRAGPipeline()
    if use_real_ragas is None:
        use_real_ragas = _real_ragas_enabled()
    if use_real_ragas and cloud_client is None:
        cloud_client = cloud_ragas.client_from_config(cloud_ragas.config_from_env())
    summaries: dict[str, dict[str, float]] = {}
    summary_backends: dict[str, str] = {}
    cloud_statuses: dict[str, str] = {}
    cloud_errors: dict[str, str] = {}
    latency_summaries: dict[str, dict[str, float]] = {}
    detail_rows: list[dict[str, Any]] = []

    rows_by_strategy: dict[str, list[dict[str, Any]]] = {}
    for index, strategy in enumerate(STRATEGIES):
        if progress_callback:
            progress_callback(index / max(len(STRATEGIES), 1), f"Evaluating {strategy}...")
        logger.info("Evaluating strategy: %s", strategy)
        summary, rows = evaluate_strategy(dataset, strategy, pipeline)
        rows_by_strategy[strategy] = rows
        summaries[strategy] = summary
        summary_backends[strategy] = "offline_heuristic"
        cloud_statuses[strategy] = "not_requested"
        cloud_errors[strategy] = ""
        latency_summaries[strategy] = _latency_summary(rows)
        detail_rows.extend(rows)
        logger.info("Strategy %s: %s (backend=%s)", strategy, summary, "offline_heuristic")

    if use_real_ragas and cloud_client is not None:
        for index, strategy in enumerate(STRATEGIES):
            if progress_callback:
                progress_callback(index / max(len(STRATEGIES), 1), f"Cloud RAGAS: {strategy}...")
            real_scores, cloud_error = maybe_run_real_ragas_with_status(rows_by_strategy.get(strategy, []), cloud_client=cloud_client, enabled=True)
            if real_scores:
                summaries[strategy] = real_scores
                summary_backends[strategy] = "cloud_free_tier_ragas"
                cloud_statuses[strategy], cloud_errors[strategy] = _cloud_ragas_score_status(real_scores)
            else:
                cloud_statuses[strategy] = "fallback_offline"
                cloud_errors[strategy] = cloud_error or "Cloud RAGAS did not return scores for this strategy."

    for row in detail_rows:
        row["summary_backend"] = summary_backends.get(row["strategy"], "offline_heuristic")
    if progress_callback:
        progress_callback(1.0, "Evaluation complete.")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    current_source = load_current_source()
    evaluated_source = current_source.get("source_slug", "") if current_source else ""
    summary_frame = pd.DataFrame.from_dict(summaries, orient="index")
    summary_frame["summary_backend"] = pd.Series(summary_backends)
    summary_frame["cloud_status"] = pd.Series(cloud_statuses)
    summary_frame["cloud_error"] = pd.Series(cloud_errors)
    summary_frame["evaluated_source"] = evaluated_source
    for metric in LATENCY_SUMMARY_METRICS:
        summary_frame[metric] = pd.Series(
            {
                strategy: values.get(metric)
                for strategy, values in latency_summaries.items()
            }
        )
    summary_frame.to_csv(RAGAS_RESULTS_PATH, index_label="strategy")
    pd.DataFrame(detail_rows).to_csv(RAGAS_PER_QUESTION_PATH, index=False)
    print_markdown_report(summaries)
    return summaries


def build_index():
    from ingestion import build_index as _build_index

    return _build_index()


def print_markdown_report(results: dict[str, dict[str, float]]) -> None:
    has_ragas = any(set(scores) & set(RAGAS_METRICS) for scores in results.values())
    has_lexical = any(set(scores) & set(LEXICAL_METRICS) for scores in results.values())
    metrics = [*(RAGAS_METRICS if has_ragas else []), *(LEXICAL_METRICS if has_lexical else [])]
    if not metrics:
        metrics = RAGAS_METRICS
    print("| strategy | " + " | ".join(metrics) + " |")
    print("|---|" + "|".join("---" for _ in metrics) + "|")
    for strategy, scores in results.items():
        values = " | ".join(_format_metric_value(scores.get(metric)) for metric in metrics)
        print(f"| {strategy} | {values} |")

    averages = {
        strategy: sum(values) / len(values)
        for strategy, scores in results.items()
        if (values := [float(value) for value in scores.values() if isinstance(value, (int, float)) and math.isfinite(float(value))])
    }
    if not averages:
        print("\nBest strategy: n/a (no finite metric values were produced).")
        return
    best = max(averages, key=averages.get)
    worst_strategy, worst_metric, worst_value = min(
        (
            (strategy, metric, value)
            for strategy, scores in results.items()
            for metric, value in scores.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ),
        key=lambda item: item[2],
    )
    print(f"\nBest strategy: {best} (highest mean metric score; verify against per-question rows).")
    print(
        f"Worst metric: {worst_metric} on {worst_strategy} = {worst_value:.3f}. "
        "Hypothesis: retrieved context or extractive answer lacks enough lexical overlap with the reference."
    )


def _format_metric_value(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.3f}"


def _load_indexed_pipeline() -> LocalRAGPipeline | None:
    if load_current_source() is None:
        return None
    pipeline = LocalRAGPipeline(allow_index_build=False)
    pipeline._ensure_ready()
    return pipeline if pipeline.index is not None and pipeline.nodes else None


def generate_pre_questions(
    output_path: Path = GOLDEN_DATASET_PATH,
    providers: Sequence[QuestionProvider] | None = None,
    chunk_limit: int = 12,
    final_limit: int = 8,
) -> list[dict[str, str]]:
    pipeline = _load_indexed_pipeline()
    nodes: Sequence[Any] | None = pipeline.nodes if pipeline is not None else None
    if nodes is None:
        nodes = load_local_context_nodes()
    if not nodes:
        raise RuntimeError("No indexed or local source context was found for question generation.")
    invalidate_golden_dataset_if_stale()
    return generate_golden_dataset(
        nodes,
        output_path=output_path,
        providers=providers,
        chunk_limit=chunk_limit,
        final_limit=final_limit,
    )


def main(use_real_ragas: bool | None = None) -> None:
    pipeline = _load_indexed_pipeline()
    nodes: Sequence[Any] | None = None
    if use_real_ragas is None:
        use_real_ragas = _real_ragas_enabled()
    cloud_client = (
        cloud_ragas.client_from_config(cloud_ragas.config_from_env()) if use_real_ragas else None
    )
    invalidate_golden_dataset_if_stale()
    try:
        load_golden_dataset(GOLDEN_DATASET_PATH)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        if pipeline is not None:
            nodes = pipeline.nodes
        elif _index_build_enabled():
            index, nodes = build_index()
            pipeline = LocalRAGPipeline(index=index, nodes=nodes, allow_index_build=False)
        else:
            nodes = load_local_context_nodes()
            if not nodes:
                raise RuntimeError(
                    "Golden dataset is missing or invalid, and no local context files were found. "
                    "Add files under data/raw or data/eval, or set ALLOW_INDEX_BUILD=1 to allow "
                    "scraping/model setup explicitly."
                )
        if cloud_client:
            generate_golden_dataset(
                nodes,
                output_path=GOLDEN_DATASET_PATH,
                providers=default_question_providers(cloud_client=cloud_client),
            )
        else:
            generate_golden_dataset(nodes, output_path=GOLDEN_DATASET_PATH)

    pipeline = pipeline or LocalRAGPipeline(nodes=nodes, allow_index_build=False)
    if cloud_client:
        run_evaluation(pipeline=pipeline, cloud_client=cloud_client, use_real_ragas=use_real_ragas)
    else:
        run_evaluation(pipeline=pipeline, use_real_ragas=use_real_ragas)


if __name__ == "__main__":
    main()
