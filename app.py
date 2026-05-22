"""Streamlit dashboard for offline-safe RAG querying and evaluation review."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, cast

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st: Any | None = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
QUERY_LOG_PATH = PROJECT_ROOT / "data" / "query_log.jsonl"
RAGAS_RESULTS_PATH = EVAL_DIR / "ragas_results.csv"
RAGAS_PER_QUESTION_PATH = EVAL_DIR / "ragas_per_question.csv"
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
STRATEGIES = ["semantic_only", "bm25_only", "hybrid_no_rerank", "hybrid_rerank"]
CHAT_MESSAGES_KEY = "chat_messages"
PREPARED_SOURCE_KEY = "prepared_source"
MODEL_INFO = {
    "retrieval": "local hybrid retrieval with index build enabled by default; set ALLOW_INDEX_BUILD=0 to disable",
    "cloud_chat": "generative chat enabled by default when a free-tier provider key is configured; set ALLOW_CLOUD_CHAT=0 to force extractive fallback",
    "offline_answering": "extractive fallback with visible synthesis status",
}
PER_QUESTION_COLUMNS = [
    "strategy",
    "question",
    "answer",
    "ground_truth",
    "source_doc",
    "evaluation_backend",
    "summary_backend",
    *METRICS,
]
SENSITIVE_TRACE_KEYS = {"message", "retrieval_query"}
ALLOWED_SYNTHESIS_KEYS = {
    "mode",
    "code",
    "schema_version",
    "provider_chain",
    "provider_timeout_seconds",
    "total_timeout_seconds",
    "provider_attempts",
    "synthesis_error",
}
ALLOWED_SYNTHESIS_ERROR_KEYS = {"code", "stage", "retryable", "provider"}
ALLOWED_PROVIDER_ATTEMPT_KEYS = {"provider", "model", "attempt", "outcome", "status_code", "duration_ms", "error_class"}


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return _empty_frame(columns)


def _coerce_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for metric in METRICS:
        if metric not in frame.columns:
            frame[metric] = pd.NA
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame


def load_eval_summary(path: Path = RAGAS_RESULTS_PATH) -> pd.DataFrame:
    frame = _read_csv(path, ["strategy", *METRICS, "summary_backend", "evaluated_source"])
    if "strategy" not in frame.columns:
        frame.insert(0, "strategy", None)
    if "summary_backend" not in frame.columns:
        frame["summary_backend"] = "unknown"
    if "evaluated_source" not in frame.columns:
        frame["evaluated_source"] = ""
    frame = _coerce_metric_columns(frame)
    return frame[["strategy", *METRICS, "summary_backend", "evaluated_source"]]


def load_per_question(path: Path = RAGAS_PER_QUESTION_PATH) -> pd.DataFrame:
    frame = _read_csv(path, PER_QUESTION_COLUMNS)
    for column in PER_QUESTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = _coerce_metric_columns(frame)
    return frame


def best_strategy(summary: pd.DataFrame) -> str | None:
    if summary.empty or "strategy" not in summary.columns:
        return None

    scored = summary[["strategy", *METRICS]].copy()
    scored["mean_score"] = scored[METRICS].mean(axis=1, skipna=True)
    scored = scored.dropna(subset=["strategy", "mean_score"])
    if scored.empty:
        return None
    return str(scored.sort_values("mean_score", ascending=False).iloc[0]["strategy"])


def metric_card_values(summary: pd.DataFrame, strategy: str | None = None) -> dict[str, Any]:
    selected_strategy = strategy or best_strategy(summary)
    values: dict[str, Any] = {"strategy": selected_strategy}
    if not selected_strategy:
        return values | {metric: None for metric in METRICS}

    rows = summary[summary["strategy"] == selected_strategy]
    if rows.empty:
        return values | {metric: None for metric in METRICS}

    row = rows.iloc[0]
    for metric in METRICS:
        value = row.get(metric)
        values[metric] = None if pd.isna(value) else float(value)
    return values


def filter_questions_below_threshold(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    available_metrics = [metric for metric in METRICS if metric in frame.columns]
    if frame.empty or not available_metrics:
        return frame.copy()
    mask = frame[available_metrics].lt(threshold).any(axis=1)
    return frame[mask].copy()


def eval_backend_counts(summary: pd.DataFrame, per_question: pd.DataFrame) -> dict[str, dict[str, int]]:
    summary_counts: dict[str, int] = {}
    question_counts: dict[str, int] = {}
    if "summary_backend" in summary.columns:
        summary_counts = {str(k): v for k, v in summary["summary_backend"].dropna().astype(str).value_counts().items()}
    if "evaluation_backend" in per_question.columns:
        question_counts = {str(k): v for k, v in per_question["evaluation_backend"].dropna().astype(str).value_counts().items()}
    return {"summary_backends": summary_counts, "question_backends": question_counts}


def _score_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        if "score" in value:
            return [_normalize_score_row(value)]
        return [
            {"source": str(source), "score": score}
            for source, score in value.items()
        ]
    if isinstance(value, list):
        return [_normalize_score_row(item) for item in value]
    return []


def _normalize_score_row(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        source = item.get("source") or item.get("source_doc") or item.get("doc") or item.get("id")
        score = item.get("score")
        return {"source": source, "score": score}
    if isinstance(item, tuple) and len(item) >= 2:
        return {"source": item[0], "score": item[1]}
    return {"source": str(item), "score": None}


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sanitize_provider_attempts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    attempts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        attempts.append({key: item[key] for key in ALLOWED_PROVIDER_ATTEMPT_KEYS if key in item})
    return attempts


def _sanitize_synthesis_trace(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    sanitized: dict[str, Any] = {}
    for key in ALLOWED_SYNTHESIS_KEYS:
        if key not in value:
            continue
        if key == "provider_attempts":
            sanitized[key] = _sanitize_provider_attempts(value.get(key))
        elif key == "synthesis_error" and isinstance(value.get(key), dict):
            sanitized[key] = {
                inner_key: value[key][inner_key]
                for inner_key in ALLOWED_SYNTHESIS_ERROR_KEYS
                if inner_key in value[key]
            }
        else:
            sanitized[key] = value[key]
    return sanitized


def _sanitize_trace_metadata(trace: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in trace.items():
        if key in SENSITIVE_TRACE_KEYS:
            continue
        if key == "synthesis":
            metadata[key] = _sanitize_synthesis_trace(value)
            continue
        metadata[key] = value
    return metadata


def _query_log_entry(query: str, strategy: str, result: dict[str, Any]) -> dict[str, Any]:
    import datetime

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query_hash": _hash_text(query),
        "answer_hash": _hash_text(str(result.get("answer", ""))),
        "strategy": strategy,
        "sources": [s.get("source_doc", "") for s in result.get("sources", [])],
        "contexts_count": len(result.get("contexts", [])),
        "trace": normalize_trace(result.get("trace") if isinstance(result.get("trace"), dict) else None),
        "current_source": (load_current_source() or {}).get("source_slug", ""),
    }


def normalize_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    trace = trace or {}
    bm25 = trace.get("bm25_scores") or trace.get("bm25_results")
    vector = trace.get("vector_scores") or trace.get("vector_results") or trace.get("semantic_scores")
    rrf = trace.get("rrf_scores") or trace.get("fused_scores") or trace.get("fusion_scores")
    reranker = trace.get("reranker_scores") or trace.get("rerank_scores") or trace.get("reranked_results")
    score_keys = {
        "bm25_scores",
        "bm25_results",
        "vector_scores",
        "vector_results",
        "semantic_scores",
        "rrf_scores",
        "fused_scores",
        "fusion_scores",
        "reranker_scores",
        "rerank_scores",
        "reranked_results",
    }
    return {
        "bm25_scores": _score_rows(bm25),
        "vector_scores": _score_rows(vector),
        "rrf_scores": _score_rows(rrf),
        "reranker_scores": _score_rows(reranker),
        "metadata": _sanitize_trace_metadata({key: value for key, value in trace.items() if key not in score_keys}),
    }


def dataset_stats(golden_path: Path = GOLDEN_DATASET_PATH, per_question_path: Path = RAGAS_PER_QUESTION_PATH) -> dict[str, Any]:
    stats = {"golden_questions": 0, "evaluated_rows": 0}
    try:
        data = json.loads(golden_path.read_text(encoding="utf-8"))
        stats["golden_questions"] = len(data) if isinstance(data, list) else 0
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    stats["evaluated_rows"] = len(load_per_question(per_question_path))
    return stats


CURRENT_SOURCE_PATH = PROJECT_ROOT / "data" / "current_source.json"


def load_current_source(path: Path = CURRENT_SOURCE_PATH) -> dict[str, Any] | None:
    """Load data/current_source.json; return None if missing."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_golden_dataset_stale(current_source: dict | None, golden_path: Path = GOLDEN_DATASET_PATH) -> bool:
    """Check if golden dataset was generated for a different source than the current one."""
    if current_source is None:
        return False
    if not golden_path.exists():
        return False
    try:
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(golden, list) or not golden:
        return False
    first = golden[0]
    if not isinstance(first, dict):
        return False
    golden_slug = first.get("source_slug")
    return golden_slug != current_source.get("source_slug")


def log_query(query: str, strategy: str, result: dict[str, Any], log_path: Path = QUERY_LOG_PATH) -> None:
    """Append a query interaction to the JSONL log file."""
    entry = _query_log_entry(query, strategy, result)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Query logged: hash=%s (strategy=%s)", entry["query_hash"][:12], strategy)


def _has_raw_source_files(raw_dir: Path = RAW_DIR) -> bool:
    """Return True if data/raw/ has any files (recursively)."""
    if not raw_dir.exists():
        return False
    return any(raw_dir.rglob("*"))


def get_source_badge_state(
    current_source_path: Path = CURRENT_SOURCE_PATH,
    chroma_dir: Path = CHROMA_DIR,
    raw_dir: Path = RAW_DIR,
) -> str:
    """Return 'green' if indexed, 'yellow' if prepared, 'grey' otherwise."""
    has_source = current_source_path.exists()
    has_chroma = chroma_dir.exists()
    has_raw = _has_raw_source_files(raw_dir)

    if has_source and has_chroma:
        return "green"
    if has_raw:
        return "yellow"
    return "grey"


def run_build_index(source_files: Iterable[Path] | None = None) -> list:
    """Build the vector index from data/raw/ sources. Returns the nodes list."""
    from ingestion import build_index

    files = list(source_files) if source_files is not None else None
    raw_dir = files[0].parent if files else RAW_DIR
    _index, nodes = build_index(source_files=files, raw_dir=raw_dir)
    return nodes


def run_evaluation_inline() -> None:
    """Run full evaluation pipeline inline (blocking). Calls evaluate.main()."""
    from evaluate import main as eval_main

    eval_main()


def last_eval_date(path: Path = RAGAS_RESULTS_PATH) -> str:
    if not path.exists():
        return "Not available"
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")


def build_grouped_bar_chart(summary: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if summary.empty:
        ax.text(0.5, 0.5, "No evaluation data", ha="center", va="center")
        ax.axis("off")
        return fig

    summary.set_index("strategy")[METRICS].plot(kind="bar", ax=ax)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def run_query(query: str, strategy: str) -> dict[str, Any]:
    from pipeline import answer_query

    return answer_query(query, strategy=strategy, allow_index_build=True)


def run_chat_query(message: str, history: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    from pipeline import chat_query

    return chat_query(message, history=history, strategy=strategy, allow_index_build=True)


def prepare_sources_for_app(
    sources: list[str],
    allow_github_fetch: bool = False,
    allow_huggingface_fetch: bool = False,
) -> list[Path]:
    from source_loader import prepare_sources

    cleaned = [source.strip() for source in sources if source.strip()]
    if not cleaned:
        return []
    return prepare_sources(
        cast(list[str | Path], cleaned),
        allow_github_fetch=allow_github_fetch,
        allow_huggingface_fetch=allow_huggingface_fetch,
    )


def _prepared_source_state(st) -> dict[str, Any] | None:
    value = st.session_state.get(PREPARED_SOURCE_KEY)
    return value if isinstance(value, dict) else None


def _current_source_for_ui(st) -> dict[str, Any] | None:
    prepared = _prepared_source_state(st)
    indexed = load_current_source()
    if prepared and prepared.get("source_slug") != (indexed or {}).get("source_slug"):
        return prepared
    return indexed


def _source_is_indexed(current_source: dict[str, Any] | None) -> bool:
    if current_source is None or not CHROMA_DIR.exists():
        return False
    if current_source.get("indexed_at"):
        indexed = load_current_source()
        if indexed is None:
            return True
        return current_source.get("source_slug") == indexed.get("source_slug")
    return False


def _active_indexed_source(st) -> dict[str, Any] | None:
    current_source = _current_source_for_ui(st)
    return current_source if _source_is_indexed(current_source) else None


def _active_prepared_only_source(st) -> dict[str, Any] | None:
    current_source = _current_source_for_ui(st)
    if current_source is None:
        return None
    return None if _source_is_indexed(current_source) else current_source


def _pending_index_message(current_source: dict[str, Any]) -> str:
    return (
        f"Source '{current_source.get('source_slug', 'unknown')}' is prepared but not indexed yet. "
        "Build the index in the Sources tab first."
    )


def _degraded_query_message(current_source: dict[str, Any]) -> str:
    return (
        f"Source '{current_source.get('source_slug', 'unknown')}' is prepared but not indexed yet. "
        "Build the index in the Sources tab before querying. "
        "Query will fall back to lexical-only search with degraded quality."
    )


def _sources_status_message(current_source: dict[str, Any]) -> str:
    return f"Prepared source pending index: {current_source.get('source_slug', 'unknown')}"


def _indexed_status_message(current_source: dict[str, Any]) -> str:
    return f"Current index: {current_source.get('source_slug', 'unknown')}"


def _indexed_caption(current_source: dict[str, Any]) -> str:
    return f"Indexed source: {current_source.get('source_slug', 'unknown')}"


def _evaluated_info(current_source: dict[str, Any]) -> str:
    return f"Evaluated on: {current_source.get('source_slug', 'unknown')} | Indexed: {current_source.get('indexed_at', 'unknown')}"


def _last_eval_caption(evaluated_source: str, question_count: int, eval_date: str) -> str:
    return f"Last eval: {evaluated_source} | {question_count} questions | {eval_date}"


def _source_prepared(current_source: dict | None) -> bool:
    return current_source is not None and not _source_is_indexed(current_source)


def _can_query(current_source: dict | None) -> bool:
    return _source_is_indexed(current_source)


def _can_evaluate(current_source: dict | None) -> bool:
    return _source_is_indexed(current_source)


def _has_any_indexed_source() -> bool:
    indexed = load_current_source()
    return indexed is not None and CHROMA_DIR.exists()


def _source_for_eval_staleness(st) -> dict | None:
    return _active_indexed_source(st)


def _source_for_eval_header(st) -> dict | None:
    return _active_indexed_source(st)


def _source_for_query(st) -> dict | None:
    return _active_indexed_source(st)


def _source_for_prepared_warning(st) -> dict | None:
    return _active_prepared_only_source(st)


def _source_for_sources_tab(st) -> dict | None:
    return _current_source_for_ui(st)


def _source_for_sidebar(st) -> dict | None:
    return _current_source_for_ui(st)


def _source_for_eval_run(st) -> dict | None:
    return _active_indexed_source(st)


def _source_for_eval_pending(st) -> dict | None:
    return _active_prepared_only_source(st)


def _source_for_query_pending(st) -> dict | None:
    return _active_prepared_only_source(st)


def _has_index_for_current_source(st) -> bool:
    return _active_indexed_source(st) is not None


def _show_index_required_error(st, current_source: dict[str, Any]) -> None:
    st.error(_pending_index_message(current_source))


def _show_query_warning(st, current_source: dict[str, Any]) -> None:
    st.warning(_degraded_query_message(current_source))


def _show_eval_warning(st, current_source: dict[str, Any]) -> None:
    st.warning(_pending_index_message(current_source))


def _show_sources_pending_info(st, current_source: dict[str, Any]) -> None:
    st.info(_sources_status_message(current_source))


def _show_sources_index_info(st, current_source: dict[str, Any]) -> None:
    st.info(_indexed_status_message(current_source))


def _show_query_caption(st, current_source: dict[str, Any]) -> None:
    st.caption(_indexed_caption(current_source))


def _show_eval_header(st, current_source: dict[str, Any]) -> None:
    st.info(_evaluated_info(current_source))


def _show_last_eval_caption(st, evaluated_source: str, question_count: int, eval_date: str) -> None:
    st.caption(_last_eval_caption(evaluated_source, question_count, eval_date))


def _has_indexed_source_for_eval(st) -> bool:
    return _active_indexed_source(st) is not None


def _prepared_source_for_eval(st) -> dict | None:
    return _active_prepared_only_source(st)


def _prepared_source_for_query(st) -> dict | None:
    return _active_prepared_only_source(st)


def _indexed_source_for_query(st) -> dict | None:
    return _active_indexed_source(st)


def _indexed_source_for_eval(st) -> dict | None:
    return _active_indexed_source(st)


def _show_general_no_source_warning(st) -> None:
    st.warning(
        "No source is indexed yet. Go to the Sources tab to prepare and index a source. "
        "Query will fall back to lexical-only search with degraded quality."
    )


def _show_general_no_eval_source_warning(st) -> None:
    st.warning("No source is indexed yet. Go to the Sources tab to prepare and index a source before running evaluation.")


def _show_general_no_eval_source_error(st) -> None:
    st.error("Cannot run evaluation: no source is indexed. Go to the Sources tab first.")


def _show_allow_index_build_error(st) -> None:
    st.error(
        "Index build is disabled because ALLOW_INDEX_BUILD=0. "
        "Unset it or set ALLOW_INDEX_BUILD=1 and restart the app."
    )


def _has_indexed_source_matching_current(st) -> bool:
    return _active_indexed_source(st) is not None


def _show_stale_dataset_warning(st, current_source: dict[str, Any]) -> None:
    st.warning(
        f"The golden dataset was generated for a different source than the currently indexed one "
        f"('{current_source.get('source_slug', '?')}'). Click 'Run Evaluation' to regenerate."
    )


def _show_stale_eval_results_warning(st, current_source: dict[str, Any]) -> None:
    st.warning(
        f"The saved evaluation results were generated for a different source than the currently indexed one "
        f"('{current_source.get('source_slug', '?')}'). Click 'Run Evaluation' to regenerate."
    )


def _show_no_raw_files_error(st) -> None:
    st.error("No source files found in data/raw/. Prepare a source first.")


def _show_eval_success(st) -> None:
    st.success("Evaluation complete. Refreshing...")


def _show_build_success(st, nodes: list, slug: str) -> None:
    st.success(f"Index built successfully. {len(nodes)} chunks for source '{slug}'.")


def _show_build_failure(st, exc: Exception) -> None:
    st.error(f"Index build failed: {exc}")


def _show_eval_failure(st, exc: Exception) -> None:
    st.error(f"Evaluation failed: {exc}")


def _show_prepare_failure(st, exc: Exception) -> None:
    st.error(str(exc))


def _show_prepare_empty(st) -> None:
    st.warning("No sources were provided.")


def _show_prepare_success(st, prepared: list[Path]) -> None:
    st.success(f"Prepared {len(prepared)} files under data/raw.")


def _store_prepared_source(st, prepared: list[Path], source_input: str, source_type: str) -> None:
    st.session_state["prepared_files"] = prepared
    st.session_state[PREPARED_SOURCE_KEY] = {
        "source_slug": prepared[0].parent.name,
        "source_input": source_input,
        "source_type": source_type,
        "indexed_at": None,
    }


def _clear_prepared_source(st) -> None:
    st.session_state.pop(PREPARED_SOURCE_KEY, None)


def _prepared_source_type(source_type: str) -> str:
    return "huggingface" if source_type == "HuggingFace model/dataset" else "github" if source_type == "GitHub repo" else "local"


def _render_prepared_files(st, prepared: list[Path]) -> None:
    st.dataframe(
        pd.DataFrame({"file": [path.as_posix() for path in prepared]}),
        use_container_width=True,
    )


def _prepared_source_slug(prepared: list[Path]) -> str:
    return prepared[0].parent.name


def _show_prepared_source_info(st, prepared: list[Path]) -> None:
    st.info(f"Prepared source: {_prepared_source_slug(prepared)}")


def _current_or_prepared_source(st) -> dict | None:
    return _current_source_for_ui(st)


def _current_or_prepared_source_is_indexed(st) -> bool:
    return _source_is_indexed(_current_source_for_ui(st))


def _prepared_source_exists(st) -> bool:
    return _active_prepared_only_source(st) is not None


def _indexed_source_exists(st) -> bool:
    return _active_indexed_source(st) is not None


def _current_index_slug(st) -> str | None:
    source = _active_indexed_source(st)
    return source.get("source_slug") if source else None


def _prepared_index_slug(st) -> str | None:
    source = _active_prepared_only_source(st)
    return source.get("source_slug") if source else None


def _allow_index_build_enabled() -> bool:
    return os.getenv("ALLOW_INDEX_BUILD", "1") != "0"


def _has_prepared_files(st) -> bool:
    return "prepared_files" in st.session_state or _has_raw_source_files()


def _build_index_ready(st) -> bool:
    return _has_prepared_files(st)


def _source_status_for_sidebar(st) -> tuple[str, dict[str, Any] | None]:
    return get_source_badge_state(), _source_for_sidebar(st)


def _render_sidebar_status(st) -> None:
    badge_state, current_source = _source_status_for_sidebar(st)
    _render_source_badge(st, badge_state, current_source)


def _render_query_unavailable(st, current_source: dict[str, Any] | None) -> None:
    if current_source is None:
        _show_general_no_source_warning(st)
        return
    if _source_prepared(current_source):
        _show_query_warning(st, current_source)
    else:
        _show_general_no_source_warning(st)


def _render_eval_unavailable(st, current_source: dict[str, Any] | None) -> None:
    if current_source is None:
        _show_general_no_eval_source_warning(st)
        return
    if _source_prepared(current_source):
        _show_eval_warning(st, current_source)
    else:
        _show_general_no_eval_source_warning(st)


def _run_evaluation_available(st, current_source: dict[str, Any] | None) -> None:
    if current_source is None:
        _show_general_no_eval_source_error(st)
        return
    if _source_prepared(current_source):
        _show_index_required_error(st, current_source)
    else:
        _show_general_no_eval_source_error(st)


def _source_eval_caption_available(summary: pd.DataFrame) -> bool:
    return not summary.empty and "evaluated_source" in summary.columns


def _source_eval_caption_values(summary: pd.DataFrame) -> tuple[str, int, str] | None:
    if summary.empty or "evaluated_source" not in summary.columns:
        return None
    evaluated_source = summary["evaluated_source"].iloc[0]
    if not evaluated_source:
        return None
    question_count = dataset_stats().get("golden_questions", 0)
    eval_date = last_eval_date()
    return str(evaluated_source), question_count, eval_date


def _eval_summary_matches_source(summary: pd.DataFrame, current_source: dict[str, Any] | None) -> bool:
    if summary.empty or current_source is None or "evaluated_source" not in summary.columns:
        return True
    evaluated_sources = summary["evaluated_source"].dropna().astype(str)
    evaluated_sources = evaluated_sources[evaluated_sources != ""]
    if evaluated_sources.empty:
        return True
    return set(evaluated_sources) == {str(current_source.get("source_slug", ""))}


def _discard_stale_eval_results(
    summary: pd.DataFrame,
    per_question: pd.DataFrame,
    current_source: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    if _eval_summary_matches_source(summary, current_source):
        return summary, per_question, False
    return _empty_frame(["strategy", *METRICS, "summary_backend", "evaluated_source"]), _empty_frame(PER_QUESTION_COLUMNS), True


def _show_eval_caption_if_available(st, summary: pd.DataFrame) -> None:
    values = _source_eval_caption_values(summary)
    if values is not None:
        _show_last_eval_caption(st, *values)


def _show_eval_header_if_available(st, current_source: dict[str, Any] | None) -> None:
    if current_source is not None:
        _show_eval_header(st, current_source)


def _show_sources_status_info(st, current_source: dict[str, Any] | None) -> None:
    if current_source is not None:
        if _source_is_indexed(current_source):
            _show_sources_index_info(st, current_source)
        else:
            _show_sources_pending_info(st, current_source)


def _render_query_caption_if_available(st, current_source: dict[str, Any] | None) -> None:
    if current_source is not None and _source_is_indexed(current_source):
        _show_query_caption(st, current_source)


def _active_current_source(st) -> dict[str, Any] | None:
    return _current_source_for_ui(st)


def _active_current_source_indexed(st) -> dict[str, Any] | None:
    return _active_indexed_source(st)


def _active_current_source_prepared(st) -> dict[str, Any] | None:
    return _active_prepared_only_source(st)


def _needs_query_block(st) -> bool:
    return not _has_index_for_current_source(st)


def _needs_eval_block(st) -> bool:
    return not _has_indexed_source_for_eval(st)


def _show_query_block_message(st) -> None:
    current_source = _active_current_source_prepared(st)
    if current_source is not None:
        _show_query_warning(st, current_source)
    else:
        _show_general_no_source_warning(st)


def _show_eval_block_message(st) -> None:
    current_source = _active_current_source_prepared(st)
    if current_source is not None:
        _show_eval_warning(st, current_source)
    else:
        _show_general_no_eval_source_warning(st)


def _show_eval_run_block_message(st) -> None:
    current_source = _active_current_source_prepared(st)
    if current_source is not None:
        _show_index_required_error(st, current_source)
    else:
        _show_general_no_eval_source_error(st)


def _render_query_caption_ready(st) -> None:
    current_source = _active_current_source_indexed(st)
    if current_source is not None:
        _show_query_caption(st, current_source)


def _render_eval_header_ready(st) -> None:
    current_source = _active_current_source_indexed(st)
    if current_source is not None:
        _show_eval_header(st, current_source)


def _render_sources_status_ready(st) -> None:
    current_source = _active_current_source(st)
    _show_sources_status_info(st, current_source)


def _prepared_source_for_message(st) -> dict[str, Any] | None:
    return _active_current_source_prepared(st)


def _indexed_source_for_message(st) -> dict[str, Any] | None:
    return _active_current_source_indexed(st)


def _query_should_render_controls(st) -> bool:
    return _indexed_source_for_message(st) is not None


def _eval_should_allow_run(st) -> bool:
    return _indexed_source_for_message(st) is not None


def _maybe_show_eval_caption(st, summary: pd.DataFrame) -> None:
    _show_eval_caption_if_available(st, summary)


def _maybe_show_eval_header(st) -> None:
    _render_eval_header_ready(st)


def _maybe_show_query_caption(st) -> None:
    _render_query_caption_ready(st)


def _maybe_show_sources_status(st) -> None:
    _render_sources_status_ready(st)


def _render_query_blocked(st) -> None:
    _show_query_block_message(st)


def _render_eval_blocked(st) -> None:
    current_source = _active_current_source_prepared(st)
    if current_source is not None:
        _show_sources_pending_info(st, current_source)
    _show_eval_block_message(st)


def _render_eval_run_blocked(st) -> None:
    _show_eval_run_block_message(st)


def _render_build_index_status(st) -> None:
    _maybe_show_sources_status(st)


def _render_query_ready(st) -> None:
    _maybe_show_query_caption(st)


def _render_eval_ready(st, summary: pd.DataFrame) -> None:
    _maybe_show_eval_header(st)
    _maybe_show_eval_caption(st, summary)


def _prepared_source_session_exists(st) -> bool:
    return PREPARED_SOURCE_KEY in st.session_state


def _pending_source_slug_from_session(st) -> str | None:
    prepared = _prepared_source_state(st)
    return prepared.get("source_slug") if prepared else None


def _pending_source_matches_index(st) -> bool:
    prepared = _prepared_source_state(st)
    indexed = load_current_source()
    return bool(prepared and indexed and prepared.get("source_slug") == indexed.get("source_slug"))


def _cleanup_pending_source_if_index_matches(st) -> None:
    if _pending_source_matches_index(st):
        _clear_prepared_source(st)


def _initialize_current_source_ui_state(st) -> None:
    _cleanup_pending_source_if_index_matches(st)


def _source_ui_state(st) -> dict[str, Any] | None:
    _initialize_current_source_ui_state(st)
    return _current_source_for_ui(st)


def _render_source_badge(st, badge_state: str, current_source: dict[str, Any] | None) -> None:
    """Render a colored source status badge in the sidebar."""
    if badge_state == "green":
        slug = current_source.get("source_slug", "unknown") if current_source else "unknown"
        indexed_at = current_source.get("indexed_at", "unknown") if current_source else "unknown"
        st.sidebar.success(f"Source indexed: {slug}")
        st.sidebar.caption(f"Indexed at: {indexed_at}")
    elif badge_state == "yellow":
        slug = current_source.get("source_slug", "unknown") if current_source else "unknown"
        st.sidebar.warning(f"Source prepared but not indexed: {slug}")
    else:
        st.sidebar.error("No source prepared")


def _render_sidebar(st) -> None:
    current_source = _source_ui_state(st)
    badge_state = get_source_badge_state()
    _render_source_badge(st, badge_state, current_source)

    stats = dataset_stats()
    st.sidebar.header("Run metadata")
    st.sidebar.write("Model")
    st.sidebar.json(MODEL_INFO)
    st.sidebar.write("Dataset stats")
    st.sidebar.json(stats)
    st.sidebar.write(f"Last eval: {last_eval_date()}")

    data = RAGAS_RESULTS_PATH.read_bytes() if RAGAS_RESULTS_PATH.exists() else b""
    st.sidebar.download_button(
        "Download results CSV",
        data=data,
        file_name="ragas_results.csv",
        mime="text/csv",
        disabled=not bool(data),
    )


def _render_sources_tab(st) -> None:
    st.subheader("Source preparation")

    source_type = st.radio(
        "Source type",
        ["Local directory", "GitHub repo", "HuggingFace model/dataset"],
    )

    # Adaptive input field
    if source_type == "Local directory":
        source_text = st.text_input("Local path", placeholder="C:\\path\\to\\project")
    elif source_type == "GitHub repo":
        source_text = st.text_input("GitHub URL", placeholder="https://github.com/user/repo")
    else:
        source_text = st.text_input("HuggingFace URL", placeholder="hf:owner/model or https://huggingface.co/owner/model")

    allow_network = False
    if source_type in ("GitHub repo", "HuggingFace model/dataset"):
        allow_network = st.checkbox(f"Allow {source_type.split()[0].lower()} fetch")

    # Prepare sources
    if st.button("Prepare sources"):
        if not source_text or not source_text.strip():
            st.warning("No sources were provided.")
        else:
            sources = [source_text.strip()]
            try:
                if source_type == "HuggingFace model/dataset":
                    prepared = prepare_sources_for_app(
                        sources, allow_github_fetch=False, allow_huggingface_fetch=allow_network,
                    )
                elif source_type == "GitHub repo":
                    prepared = prepare_sources_for_app(sources, allow_github_fetch=allow_network)
                else:
                    prepared = prepare_sources_for_app(sources, allow_github_fetch=False, allow_huggingface_fetch=False)
            except Exception as exc:
                st.error(str(exc))
                return
            if prepared:
                st.success(f"Prepared {len(prepared)} files under data/raw.")
                st.session_state["prepared_files"] = prepared
                st.session_state[PREPARED_SOURCE_KEY] = {
                    "source_slug": prepared[0].parent.name,
                    "source_input": sources[0],
                    "source_type": "huggingface" if source_type == "HuggingFace model/dataset" else "github" if source_type == "GitHub repo" else "local",
                    "indexed_at": None,
                }
                st.dataframe(
                    pd.DataFrame({"file": [path.as_posix() for path in prepared]}),
                    use_container_width=True,
                )
                st.info(f"Prepared source: {prepared[0].parent.name}")
            else:
                st.warning("No sources were provided.")

    # Build Index button — available when files exist in data/raw/
    has_prepared = "prepared_files" in st.session_state or _has_raw_source_files()
    if has_prepared:
        current_source = _source_ui_state(st)
        _show_sources_status_info(st, current_source)

        if st.button("Build Index"):
            if not _has_raw_source_files():
                st.error("No source files found in data/raw/. Prepare a source first.")
            elif not _allow_index_build_enabled():
                _show_allow_index_build_error(st)
            else:
                progress = st.progress(0.0, "Building index...")
                try:
                    nodes = run_build_index(st.session_state.get("prepared_files"))
                    progress.progress(1.0)
                    current = load_current_source()
                    slug = current.get("source_slug", "unknown") if current else "unknown"
                    st.session_state.pop(PREPARED_SOURCE_KEY, None)
                    st.success(f"Index built successfully. {len(nodes)} chunks for source '{slug}'.")
                except Exception as exc:
                    st.error(f"Index build failed: {exc}")


def _chat_history(st) -> list[dict[str, Any]]:
    if CHAT_MESSAGES_KEY not in st.session_state:
        st.session_state[CHAT_MESSAGES_KEY] = []
    return st.session_state[CHAT_MESSAGES_KEY]


def _citation_text(citation: dict[str, Any], max_snippet_chars: int = 180) -> str:
    source = citation.get("source_doc") or citation.get("source") or "unknown source"
    snippet = str(citation.get("snippet") or "").strip()
    if len(snippet) > max_snippet_chars:
        snippet = f"{snippet[: max_snippet_chars - 1].rstrip()}..."
    score = citation.get("score")
    score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
    return f"{source}{score_text}: {snippet}" if snippet else f"{source}{score_text}"


def _render_citations(st, citations: list[dict[str, Any]]) -> None:
    if not citations:
        st.caption("No citations returned.")
        return
    for citation in citations[:5]:
        st.caption(_citation_text(citation))


def _synthesis_status_caption(synthesis: dict[str, Any] | None) -> str:
    if not isinstance(synthesis, dict):
        return "Synthesis status unavailable."
    mode = str(synthesis.get("mode") or "unknown")
    code = str(synthesis.get("code") or "unavailable")
    provider_chain = synthesis.get("provider_chain")
    providers = ", ".join(str(provider) for provider in provider_chain) if isinstance(provider_chain, list) else "none"
    label = "Generative answer" if mode == "generative" else "Extractive fallback" if mode == "extractive" else "Unknown synthesis mode"
    return f"{label}: {code}. Providers: {providers}."


def _render_trace_debug(st, trace: dict[str, Any] | None) -> None:
    with st.expander("Retrieval trace / debug", expanded=False):
        normalized = normalize_trace(trace)
        synthesis = normalized["metadata"].get("synthesis") if isinstance(normalized["metadata"], dict) else None
        st.write("Synthesis status")
        st.json(synthesis or {"mode": "unknown", "code": "unavailable"})
        st.caption(_synthesis_status_caption(synthesis))
        for label, rows in [
            ("BM25 scores", normalized["bm25_scores"]),
            ("Vector scores", normalized["vector_scores"]),
            ("RRF scores", normalized["rrf_scores"]),
            ("Reranker scores", normalized["reranker_scores"]),
        ]:
            st.write(label)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.caption("Not available in the current retriever trace.")
        st.write("Metadata")
        st.json(normalized["metadata"])


def _render_chat_message(st, message: dict[str, Any]) -> None:
    with st.chat_message(message.get("role", "assistant")):
        st.write(message.get("content", ""))
        if message.get("role") == "assistant":
            _render_citations(st, message.get("citations", []))
            _render_trace_debug(st, message.get("trace"))


def _render_query_tab(st) -> None:
    current_source = _source_ui_state(st)

    if not _query_should_render_controls(st):
        _render_query_blocked(st)
        return

    _render_query_ready(st)


    strategy = st.selectbox("Strategy", STRATEGIES, index=STRATEGIES.index("hybrid_rerank"))

    messages = _chat_history(st)
    for message in messages:
        _render_chat_message(st, message)

    prompt = st.chat_input("Ask about the indexed project context")
    if not prompt or not prompt.strip():
        return

    user_message = {"role": "user", "content": prompt.strip()}
    history = list(messages)
    messages.append(user_message)
    _render_chat_message(st, user_message)

    try:
        result = run_chat_query(prompt.strip(), history=history, strategy=strategy)
    except Exception as exc:
        st.error(str(exc))
        return

    log_query(prompt.strip(), strategy, result)

    assistant_message = {
        "role": "assistant",
        "content": result.get("answer", ""),
        "citations": result.get("citations", []),
        "trace": result.get("trace"),
    }
    messages.append(assistant_message)
    _render_chat_message(st, assistant_message)


def _render_eval_tab(st) -> None:
    current_source = _source_ui_state(st)
    indexed_source = _indexed_source_for_eval(st)
    summary = load_eval_summary()
    per_question = load_per_question()

    if not _eval_should_allow_run(st):
        _render_eval_blocked(st)
        return

    summary, per_question, stale_eval_results = _discard_stale_eval_results(summary, per_question, indexed_source)
    cards = metric_card_values(summary)
    backends = eval_backend_counts(summary, per_question)

    if indexed_source is not None and is_golden_dataset_stale(indexed_source):
        _show_stale_dataset_warning(st, indexed_source)
    if stale_eval_results and indexed_source is not None:
        _show_stale_eval_results_warning(st, indexed_source)
    _render_eval_ready(st, summary)

    if st.button("Run Evaluation"):
        if not _eval_should_allow_run(st):
            _render_eval_run_blocked(st)
        else:
            with st.spinner("Running evaluation... This may take a minute."):
                try:
                    run_evaluation_inline()
                    _show_eval_success(st)
                    st.rerun()
                except Exception as exc:
                    _show_eval_failure(st, exc)

    st.subheader("Evaluation summary")
    card_columns = st.columns(4)
    for column, metric in zip(card_columns, METRICS, strict=True):
        column.metric(metric, _format_metric(cards[metric]))

    best = cards.get("strategy")
    if best:
        st.success(f"Best strategy: {best}")
    else:
        st.warning("No strategy scores available.")

    st.write("Evaluation backend")
    st.json(backends)

    st.pyplot(build_grouped_bar_chart(summary))

    st.subheader("Per-question scores")
    threshold = st.slider("Show rows with any metric below", 0.0, 1.0, 0.5, 0.05)
    filtered = filter_questions_below_threshold(per_question, threshold)
    st.dataframe(filtered, use_container_width=True)


def render_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Advanced RAG", layout="wide")
    st.title("Advanced RAG")
    _render_sidebar(st)
    sources_tab, query_tab, eval_tab = st.tabs(["Sources", "Query interface", "Evaluation dashboard"])
    with sources_tab:
        _render_sources_tab(st)
    with query_tab:
        _render_query_tab(st)
    with eval_tab:
        _render_eval_tab(st)


if __name__ == "__main__":
    render_app()
