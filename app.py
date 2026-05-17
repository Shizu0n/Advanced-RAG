"""Streamlit dashboard for offline-safe RAG querying and evaluation review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
RAGAS_RESULTS_PATH = EVAL_DIR / "ragas_results.csv"
RAGAS_PER_QUESTION_PATH = EVAL_DIR / "ragas_per_question.csv"
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
STRATEGIES = ["semantic_only", "bm25_only", "hybrid_no_rerank", "hybrid_rerank"]
CHAT_MESSAGES_KEY = "chat_messages"
MODEL_INFO = {
    "primary_configured": "Gemini 2.5 Flash",
    "mode": "opt-in metadata only; no API call from Streamlit",
    "offline_answering": "local extractive synthesis",
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
    frame = _read_csv(path, ["strategy", *METRICS, "summary_backend"])
    if "strategy" not in frame.columns:
        frame.insert(0, "strategy", pd.NA)
    if "summary_backend" not in frame.columns:
        frame["summary_backend"] = "unknown"
    frame = _coerce_metric_columns(frame)
    return frame[["strategy", *METRICS, "summary_backend"]]


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
    summary_counts = {}
    question_counts = {}
    if "summary_backend" in summary.columns:
        summary_counts = summary["summary_backend"].dropna().astype(str).value_counts().to_dict()
    if "evaluation_backend" in per_question.columns:
        question_counts = per_question["evaluation_backend"].dropna().astype(str).value_counts().to_dict()
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
        "metadata": {key: value for key, value in trace.items() if key not in score_keys},
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

    return answer_query(query, strategy=strategy, allow_index_build=False)


def run_chat_query(message: str, history: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    from pipeline import chat_query

    return chat_query(message, history=history, strategy=strategy, allow_index_build=False)


def prepare_sources_for_app(sources: list[str], allow_github_fetch: bool = False) -> list[Path]:
    from source_loader import prepare_sources

    cleaned = [source.strip() for source in sources if source.strip()]
    if not cleaned:
        return []
    return prepare_sources(cleaned, allow_github_fetch=allow_github_fetch)


def _render_sidebar(st) -> None:
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
    source_text = st.text_area(
        "Local directories/files or public GitHub URLs",
        placeholder="C:\\path\\to\\project\nhttps://github.com/user/repo",
    )
    allow_github = st.checkbox("Allow public GitHub fetch")
    if st.button("Prepare sources"):
        sources = source_text.splitlines()
        try:
            prepared = prepare_sources_for_app(sources, allow_github_fetch=allow_github)
        except Exception as exc:
            st.error(str(exc))
            return
        if prepared:
            st.success(f"Prepared {len(prepared)} files under data/raw.")
            st.dataframe(pd.DataFrame({"file": [path.as_posix() for path in prepared]}), use_container_width=True)
        else:
            st.warning("No sources were provided.")


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


def _render_trace_debug(st, trace: dict[str, Any] | None) -> None:
    with st.expander("Retrieval trace / debug", expanded=False):
        normalized = normalize_trace(trace)
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

    assistant_message = {
        "role": "assistant",
        "content": result.get("answer", ""),
        "citations": result.get("citations", []),
        "trace": result.get("trace"),
    }
    messages.append(assistant_message)
    _render_chat_message(st, assistant_message)


def _render_eval_tab(st) -> None:
    summary = load_eval_summary()
    per_question = load_per_question()
    cards = metric_card_values(summary)
    backends = eval_backend_counts(summary, per_question)

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
