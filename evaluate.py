"""Golden dataset generation and no-cost evaluation runner."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import pandas as pd
import requests

import cloud_ragas
from pipeline import LocalRAGPipeline, load_local_context_nodes, _is_code_evidence_source, _is_readme_source


PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RAGAS_RESULTS_PATH = EVAL_DIR / "ragas_results.csv"
RAGAS_PER_QUESTION_PATH = EVAL_DIR / "ragas_per_question.csv"
CURRENT_SOURCE_PATH = PROJECT_ROOT / "data" / "current_source.json"
STRATEGIES = ["semantic_only", "bm25_only", "hybrid_no_rerank", "hybrid_rerank"]
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
LEXICAL_METRICS = ["lexical_faithfulness", "lexical_answer_relevancy", "lexical_context_recall", "lexical_context_precision"]
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


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if len(part.strip()) > 20]


@dataclass
class OfflineExtractiveQuestionProvider:
    name: str = "offline_extractive"

    def generate(self, node: Any) -> dict[str, str]:
        text = _node_text(node)
        sentences = _sentences(text)
        answer = max(sentences or [text.strip()], key=len)[:700]
        keywords = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9_]{4,}", answer)[:8]]
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


def _specificity_score(item: dict[str, str]) -> tuple[int, int]:
    question = item["question"]
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", question.lower())
    return (len(question), len(set(terms)))


def _question_source_group(node: Any) -> str:
    source = _source_doc(node)
    source_key = source.lower().replace("\\", "/")
    if source_key.endswith(("package.json", "requirements.txt", "pyproject.toml", "cargo.toml", "go.mod")):
        return "manifest"
    if _is_code_evidence_source(source):
        return "code"
    if _is_readme_source(source):
        return "readme"
    if Path(source_key).suffix.lower() in {".md", ".rst", ".txt"}:
        return "docs"
    return "other"


def _select_question_source_nodes(nodes: Sequence[Any], limit: int) -> list[Any]:
    if limit <= 0:
        return []

    grouped: dict[str, list[Any]] = {"code": [], "docs": [], "manifest": [], "readme": [], "other": []}
    for node in nodes:
        grouped.setdefault(_question_source_group(node), []).append(node)

    selected: list[Any] = []
    seen: set[str] = set()
    for group in ("code", "docs", "manifest", "readme", "other"):
        for node in grouped.get(group, []):
            key = f"{_source_doc(node)}::{getattr(node, 'node_id', id(node))}"
            if key in seen:
                continue
            selected.append(node)
            seen.add(key)
            break

    for node in nodes:
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
    grouped: dict[str, list[Any]] = {"readme": [], "manifest": [], "code": [], "docs": [], "other": []}
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


def _realistic_user_query_items(nodes: Sequence[Any]) -> list[dict[str, str]]:
    context = "\n\n".join(_node_text(node)[:1000] for node in list(nodes)[:3]).strip()
    source_doc = _source_doc(nodes[0]) if nodes else "unknown"
    return [
        {
            "question": "What stack and tools does this project use?",
            "ground_truth": "List the stack declared in project documentation and confirm it with package manifests or source evidence when available.",
            "reference_context": context,
            "source_doc": source_doc,
        },
        {
            "question": "How do I set up and run this project?",
            "ground_truth": "Use the setup and run instructions documented in the retrieved project context.",
            "reference_context": context,
            "source_doc": source_doc,
        },
        {
            "question": "What is the architecture of this project?",
            "ground_truth": "Describe the project architecture using only the retrieved project context.",
            "reference_context": context,
            "source_doc": source_doc,
        },
        {
            "question": "How should I troubleshoot common setup or runtime problems?",
            "ground_truth": "Use the troubleshooting, setup, and runtime evidence documented in the retrieved project context.",
            "reference_context": context,
            "source_doc": source_doc,
        },
    ] if context else []


def filter_best_golden_items(candidates: Sequence[dict[str, str]], limit: int = 30) -> list[dict[str, str]]:
    ranked = sorted(candidates, key=_specificity_score, reverse=True)
    filtered: list[dict[str, str]] = []
    for item in ranked[:limit]:
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

    candidates = _realistic_user_query_items([*composite_nodes, *(question_nodes or nodes)])
    for node in [*composite_nodes, *question_nodes]:
        item = generate_golden_item(node, providers)
        candidates.append(item)
    if source_slug:
        for item in candidates:
            item["source_slug"] = source_slug

    realistic = candidates[:4]
    remaining = filter_best_golden_items(candidates[4:], limit=max(final_limit - len(realistic), 0))
    dataset = (realistic + remaining)[:final_limit]
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
        result = pipeline.chat_query(item["question"], strategy=strategy)
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
        detail_rows.extend(rows)
        logger.info("Strategy %s: %s (backend=%s)", strategy, summary, "offline_heuristic")

    if use_real_ragas and cloud_client is not None:
        for index, strategy in enumerate(STRATEGIES):
            if progress_callback:
                progress_callback(index / max(len(STRATEGIES), 1), f"Cloud RAGAS: {strategy}...")
            real_scores = maybe_run_real_ragas(rows_by_strategy.get(strategy, []), cloud_client=cloud_client, enabled=True)
            if real_scores:
                summaries[strategy] = real_scores
                summary_backends[strategy] = "cloud_free_tier_ragas"

    for row in detail_rows:
        row["summary_backend"] = summary_backends.get(row["strategy"], "offline_heuristic")
    if progress_callback:
        progress_callback(1.0, "Evaluation complete.")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    current_source = load_current_source()
    evaluated_source = current_source.get("source_slug", "") if current_source else ""
    summary_frame = pd.DataFrame.from_dict(summaries, orient="index")
    summary_frame["summary_backend"] = pd.Series(summary_backends)
    summary_frame["evaluated_source"] = evaluated_source
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
