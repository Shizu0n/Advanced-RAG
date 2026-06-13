"""Streamlit dashboard for offline-safe RAG querying and evaluation review."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
from contextlib import nullcontext
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
CHAT_HISTORY_DB_PATH = PROJECT_ROOT / "data" / "chat_history.sqlite3"
RAGAS_RESULTS_PATH = EVAL_DIR / "ragas_results.csv"
RAGAS_PER_QUESTION_PATH = EVAL_DIR / "ragas_per_question.csv"
CHUNKING_ABLATION_PATH = EVAL_DIR / "chunking_ablation.csv"
EMBEDDING_COMPARISON_PATH = EVAL_DIR / "embedding_comparison.csv"
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
LEXICAL_METRICS = ["lexical_faithfulness", "lexical_answer_relevancy", "lexical_context_recall", "lexical_context_precision"]
LATENCY_METRICS = ["retrieval_ms", "synthesis_ms", "total_ms"]
LATENCY_SUMMARY_METRICS = ["avg_retrieval_ms", "avg_synthesis_ms", "avg_total_ms"]
LEXICAL_TO_RAGAS_METRIC = dict(zip(METRICS, LEXICAL_METRICS, strict=True))
CLOUD_STATUS_COLUMNS = ["cloud_status", "cloud_error"]
STRATEGIES = ["semantic_only", "bm25_only", "hybrid_no_rerank", "hybrid_rerank"]
CHAT_MESSAGES_KEY = "chat_messages"
PREPARED_SOURCE_KEY = "prepared_source"
ACTIVE_CHAT_SOURCE_KEY = "active_chat_source_slug"
UI_GATE_OVERRIDES_KEY = "ui_gate_overrides"
WORKSPACE_PAGE_KEY = "workspace_page"
SOURCE_TYPE_KEY = "source_type"
DEFAULT_SOURCE_TYPE = "Upload files"
MAX_PERSISTED_CHAT_MESSAGES = 100
UI_TOGGLE_GATES = {
    "ALLOW_HF_FETCH": {"label": "Allow Hugging Face fetch", "default": False},
    "ALLOW_GITHUB_FETCH": {"label": "Allow GitHub fetch", "default": False},
    "ALLOW_DOCS_DOWNLOAD": {"label": "Allow Python docs download", "default": False},
    "ALLOW_INDEX_BUILD": {"label": "Allow index build", "default": True},
    "ALLOW_MODEL_DOWNLOADS": {"label": "Allow model downloads", "default": False},
    "ALLOW_CLOUD_CHAT": {"label": "Allow cloud chat", "default": True},
    "USE_CLOUD_FREE_TIER_RAGAS": {"label": "Use Cloud RAGAS", "default": False},
    "ALLOW_CLOUD_FREE_TIER": {"label": "Allow cloud free tier", "default": False},
}
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
    *LEXICAL_METRICS,
    *LATENCY_METRICS,
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


def _system_light_theme_css() -> str:
    return """
        @media (prefers-color-scheme: light) {
            :root {
                --rag-bg: #f6f4ef;
                --rag-bg-soft: #ebe8df;
                --rag-surface: #ffffff;
                --rag-surface-elevated: #fbfaf7;
                --rag-border: #ddd8ce;
                --rag-border-strong: #c6c0b5;
                --rag-text: #171717;
                --rag-muted: #5d625c;
                --rag-subtle: #82877f;
                --rag-accent: #9b6b16;
                --rag-accent-2: #24766c;
                --rag-danger: #ad3333;
                --rag-shadow: 0 16px 52px rgba(45, 39, 29, 0.12);
            }
        }
    """


def _inject_app_chrome(st) -> None:
    st.markdown(
        """
        <style>
        :root {
            --rag-bg: #08090a;
            --rag-bg-soft: #101113;
            --rag-surface: #141519;
            --rag-surface-elevated: #191b20;
            --rag-border: #2a2d34;
            --rag-border-strong: #3a3f48;
            --rag-text: #f4f1ea;
            --rag-muted: #a8a29a;
            --rag-subtle: #6f746f;
            --rag-accent: #d7a545;
            --rag-accent-2: #6fc7b6;
            --rag-danger: #ec7777;
            --rag-shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
        }

        """
        + _system_light_theme_css()
        + """

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--rag-bg) !important;
            color: var(--rag-text) !important;
            overflow-x: hidden !important;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            right: 1rem;
        }

        .block-container,
        .main .block-container {
            max-width: 100%;
            padding: 1.05rem 1.35rem 8.5rem !important;
        }

        .rag-shell-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            padding: 0.35rem 0 1.05rem;
            border-bottom: 1px solid var(--rag-border);
            margin-bottom: 1rem;
        }

        .rag-title {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .rag-title h1 {
            color: var(--rag-text);
            font-size: 1.35rem;
            line-height: 1.2;
            font-weight: 690;
            letter-spacing: 0;
            margin: 0;
        }

        .rag-title p {
            color: var(--rag-muted);
            font-size: 0.86rem;
            margin: 0;
        }

        .rag-status-strip {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--rag-muted);
            font-size: 0.78rem;
            max-width: min(100%, 34rem);
            min-width: 0;
            white-space: normal;
        }

        .rag-dot {
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 999px;
            background: var(--rag-accent-2);
            box-shadow: 0 0 0 4px color-mix(in srgb, var(--rag-accent-2) 16%, transparent);
        }

        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) > [data-testid="stColumn"] {
            background: var(--rag-surface);
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            box-shadow: var(--rag-shadow);
            min-width: 0 !important;
            padding: 1rem 1rem 1.1rem;
        }

        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) > [data-testid="stColumn"]:nth-child(2) {
            background: linear-gradient(180deg, var(--rag-surface-elevated), var(--rag-surface) 34%);
            min-height: calc(100vh - 12rem);
        }

        h2, h3, .stMarkdown h2, .stMarkdown h3 {
            color: var(--rag-text) !important;
            letter-spacing: 0;
        }

        .stMarkdown, .stMarkdown p, .stCaption, label, [data-testid="stWidgetLabel"] {
            color: var(--rag-muted) !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 7px;
            border: 1px solid var(--rag-border-strong);
            background: var(--rag-surface-elevated);
            color: var(--rag-text);
            font-weight: 560;
            min-height: 2.35rem;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--rag-accent);
            color: var(--rag-text);
            background: color-mix(in srgb, var(--rag-accent) 12%, var(--rag-surface-elevated));
        }

        [data-testid="stChatMessage"] {
            background: transparent;
            border: 0;
            padding: 0.65rem 0;
        }

        [data-testid="stChatMessageContent"] {
            background: var(--rag-surface-elevated);
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
        }

        [data-testid="stChatInput"] {
            position: fixed;
            left: clamp(21rem, 25vw, 27rem);
            right: clamp(18rem, 24vw, 25rem);
            bottom: 1.05rem;
            z-index: 999;
            box-sizing: border-box;
            max-width: calc(100vw - 2rem);
            padding: 0.35rem;
            background: color-mix(in srgb, var(--rag-bg) 92%, transparent);
            border: 0;
            border-radius: 10px;
            backdrop-filter: blur(16px);
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
            overflow: hidden;
        }

        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] form,
        [data-testid="stChatInput"] div[data-baseweb="textarea"],
        [data-testid="stChatInput"] div[data-baseweb="textarea"] > div {
            box-sizing: border-box !important;
            max-width: 100% !important;
            width: 100% !important;
            min-width: 0 !important;
        }

        [data-testid="stChatInput"] textarea {
            background: var(--rag-surface-elevated) !important;
            color: var(--rag-text) !important;
            border-radius: 7px !important;
            border: 0 !important;
            box-sizing: border-box !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background: var(--rag-surface-elevated);
            border-color: var(--rag-border);
            border-radius: 7px;
            min-width: 0;
        }

        div[data-baseweb="select"] *,
        div[data-baseweb="input"] *,
        div[data-baseweb="textarea"] *,
        div[data-baseweb="radio"] *,
        .stRadio label,
        .stRadio label *,
        .stCheckbox label,
        .stCheckbox label *,
        input,
        textarea {
            color: var(--rag-text) !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: var(--rag-subtle) !important;
            opacity: 1 !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            overflow: hidden;
        }

        [data-testid="stMetric"] {
            background: var(--rag-bg-soft);
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            padding: 0.75rem;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            background: var(--rag-surface-elevated);
        }

        .stAlert {
            border-radius: 8px;
            border: 1px solid var(--rag-border);
        }

        .stAlert [data-testid="stMarkdownContainer"] *,
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
            color: var(--rag-text) !important;
        }

        .stAlert *,
        [data-testid="stMarkdownContainer"] *,
        [data-testid="stMetric"] * {
            overflow-wrap: anywhere;
        }

        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) {
            align-items: stretch;
        }

        @media (max-width: 1500px) and (min-width: 901px) {
            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) {
                display: grid !important;
                grid-template-columns: minmax(260px, 320px) minmax(0, 1fr);
                gap: 1rem !important;
            }

            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) > [data-testid="stColumn"] {
                width: 100% !important;
                flex: unset !important;
            }

            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) > [data-testid="stColumn"]:nth-child(1) {
                grid-column: 1;
                grid-row: 1 / span 2;
            }

            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) > [data-testid="stColumn"]:nth-child(2) {
                grid-column: 2;
                grid-row: 1;
                min-height: 20rem;
            }

            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) > [data-testid="stColumn"]:nth-child(3) {
                grid-column: 2;
                grid-row: 2;
            }

            [data-testid="stChatInput"] {
                position: sticky;
                left: auto;
                right: auto;
                bottom: 0.75rem;
                width: 100%;
                margin-top: 1rem;
            }
        }

        @media (max-width: 900px) {
            [data-testid="stChatInput"] {
                position: sticky;
                left: auto;
                right: auto;
                bottom: 0.75rem;
                width: 100%;
                margin-top: 1rem;
            }

            .rag-shell-header {
                align-items: flex-start;
                flex-direction: column;
            }

            .rag-status-strip {
                white-space: normal;
            }

            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) {
                display: grid !important;
                grid-template-columns: minmax(0, 1fr);
                gap: 1rem !important;
            }

            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) > [data-testid="stColumn"] {
                width: 100% !important;
                flex: unset !important;
            }

            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) > [data-testid="stColumn"]:nth-child(1) {
                grid-row: 2;
            }

            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) > [data-testid="stColumn"]:nth-child(2) {
                grid-row: 1;
                min-height: auto;
            }

            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) > [data-testid="stColumn"]:nth-child(3) {
                grid-row: 3;
            }
        }

        @media (max-width: 520px) {
            .block-container,
            .main .block-container {
                padding: 0.9rem 0.85rem 2rem !important;
            }

            .rag-title h1 {
                font-size: 1.2rem;
            }

            h2, .stMarkdown h2 {
                font-size: 1.35rem !important;
            }
        }

        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main {
            height: 100%;
            max-height: 100%;
            overflow: hidden !important;
        }

        .block-container,
        .main .block-container {
            height: 100dvh;
            max-height: 100dvh;
            overflow: hidden !important;
            padding: 0.75rem 1rem !important;
        }

        .rag-shell-header {
            flex: 0 0 auto;
            padding: 0.2rem 0 0.72rem;
            margin-bottom: 0.72rem;
        }

        .st-key-source_index_panel,
        .st-key-main_workspace_panel {
            box-sizing: border-box;
            height: calc(100dvh - 10rem) !important;
            min-height: calc(100dvh - 10rem) !important;
            max-height: calc(100dvh - 10rem) !important;
            overflow: hidden !important;
            background: var(--rag-surface);
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            box-shadow: var(--rag-shadow);
            padding: 0.82rem 0.82rem 0.9rem;
            min-width: 0;
        }

        .st-key-main_workspace_panel {
            display: flex !important;
            flex-direction: column !important;
            background: linear-gradient(180deg, var(--rag-surface-elevated), var(--rag-surface) 34%);
        }

        .st-key-source_index_panel {
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }

        .st-key-source_index_panel [data-testid="stVerticalBlock"],
        .st-key-main_workspace_panel [data-testid="stVerticalBlock"] {
            min-height: 0;
        }

        .st-key-workspace_page_nav {
            flex: 0 0 auto !important;
        }

        .st-key-workspace_page_nav [role="radiogroup"] {
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            gap: 0.45rem 0.8rem;
        }

        .st-key-workspace_page_nav label[data-testid="stWidgetLabel"] {
            display: none;
        }

        .st-key-chat_page_content,
        .st-key-eval_page_content {
            flex: 1 1 auto !important;
            box-sizing: border-box;
            height: calc(100dvh - 14.4rem) !important;
            min-height: calc(100dvh - 14.4rem) !important;
            max-height: calc(100dvh - 14.4rem) !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-right: 0.2rem;
        }

        .st-key-chat_page_content {
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
        }

        .st-key-chat_history_panel {
            flex: 1 1 auto !important;
            min-height: 0 !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding: 0.35rem 0.2rem 0.7rem 0;
        }

        .st-key-chat_composer_panel {
            flex: 0 0 auto !important;
            padding-top: 0.55rem;
            border-top: 1px solid var(--rag-border);
        }

        [data-testid="stChatInput"] {
            position: sticky !important;
            left: auto !important;
            right: auto !important;
            bottom: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0.3rem !important;
        }

        [data-testid="stChatInput"] > div {
            align-items: center !important;
        }

        [data-testid="stChatInput"] textarea {
            min-height: 2.55rem !important;
            line-height: 1.35rem !important;
            padding: 0.58rem 3.25rem 0.58rem 0.75rem !important;
            resize: none !important;
        }

        [data-testid="stChatInput"] div:has(> [data-testid="stChatInputSubmitButton"]) {
            align-items: center !important;
            bottom: 0 !important;
            display: flex !important;
            height: 100% !important;
            right: 0.48rem !important;
            top: 0 !important;
        }

        [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] {
            height: 2.15rem !important;
            transform: none !important;
            width: 2.15rem !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .st-key-source_index_panel [data-testid="stDataFrame"],
        .st-key-eval_page_content [data-testid="stDataFrame"] {
            max-height: 22rem;
            overflow: auto;
        }

        @media (max-width: 900px) {
            .block-container,
            .main .block-container {
                padding: 0.62rem 0.7rem !important;
            }

            .rag-shell-header {
                gap: 0.45rem;
                padding-bottom: 0.55rem;
                margin-bottom: 0.55rem;
            }

            .rag-title p,
            .rag-status-strip {
                font-size: 0.74rem;
            }

            .st-key-source_index_panel,
            .st-key-main_workspace_panel {
                height: auto !important;
                min-height: 0 !important;
                max-height: none !important;
            }

            .st-key-source_index_panel {
                max-height: 26dvh !important;
            }

            .st-key-main_workspace_panel {
                height: 49dvh !important;
                min-height: 49dvh !important;
                max-height: 49dvh !important;
            }

            .st-key-chat_page_content,
            .st-key-eval_page_content {
                height: calc(49dvh - 4.3rem) !important;
                min-height: calc(49dvh - 4.3rem) !important;
                max-height: calc(49dvh - 4.3rem) !important;
            }

            .st-key-chat_history_panel {
                flex: 0 1 auto !important;
                height: max(2.5rem, calc(49dvh - 20rem)) !important;
                min-height: max(2.5rem, calc(49dvh - 20rem)) !important;
                max-height: max(2.5rem, calc(49dvh - 20rem)) !important;
            }

            .st-key-chat_composer_panel {
                flex: 0 0 auto !important;
            }
        }

        @media (max-width: 520px) {
            .rag-title h1 {
                font-size: 1.05rem;
            }

            h2, .stMarkdown h2 {
                font-size: 1.12rem !important;
            }

            .st-key-source_index_panel,
            .st-key-main_workspace_panel {
                padding: 0.68rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_workbench_header(st) -> None:
    st.markdown(
        """
        <div class="rag-shell-header">
            <div class="rag-title">
                <h1>Advanced RAG</h1>
                <p>Source-aware chat, retrieval traces, and RAGAS evaluation in one workspace.</p>
            </div>
            <div class="rag-status-strip">
                <span class="rag-dot"></span>
                <span>Offline-first retrieval &middot; Free-tier cloud gates &middot; Source-scoped history</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return _empty_frame(columns)


def _coerce_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for metric in [*METRICS, *LEXICAL_METRICS]:
        if metric not in frame.columns:
            frame[metric] = pd.NA
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame


def load_eval_summary(path: Path = RAGAS_RESULTS_PATH) -> pd.DataFrame:
    columns = ["strategy", *METRICS, *LEXICAL_METRICS, "summary_backend", *CLOUD_STATUS_COLUMNS, "evaluated_source", *LATENCY_SUMMARY_METRICS]
    frame = _read_csv(path, columns)
    if "strategy" not in frame.columns:
        frame.insert(0, "strategy", None)
    if "summary_backend" not in frame.columns:
        frame["summary_backend"] = "unknown"
    has_cloud_backend = frame["summary_backend"].astype(str).eq("cloud_free_tier_ragas").any()
    if "cloud_status" not in frame.columns:
        frame["cloud_status"] = frame["summary_backend"].astype(str).map(
            lambda backend: "succeeded"
            if backend == "cloud_free_tier_ragas"
            else ("fallback_offline" if has_cloud_backend else "not_requested")
        )
    else:
        inferred_status = frame["summary_backend"].astype(str).map(
            lambda backend: "succeeded"
            if backend == "cloud_free_tier_ragas"
            else ("fallback_offline" if has_cloud_backend else "not_requested")
        )
        frame["cloud_status"] = frame["cloud_status"].fillna(inferred_status)
    if "cloud_error" not in frame.columns:
        frame["cloud_error"] = ""
    frame["cloud_error"] = frame["cloud_error"].fillna("").astype(str)
    frame.loc[
        frame["cloud_status"].astype(str).eq("fallback_offline") & frame["cloud_error"].eq(""),
        "cloud_error",
    ] = "Cloud status was not recorded for this result; offline metrics were retained."
    if "evaluated_source" not in frame.columns:
        frame["evaluated_source"] = ""
    for metric in LATENCY_SUMMARY_METRICS:
        if metric not in frame.columns:
            frame[metric] = pd.NA
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = _coerce_metric_columns(frame)
    return frame[columns]


def load_per_question(path: Path = RAGAS_PER_QUESTION_PATH) -> pd.DataFrame:
    frame = _read_csv(path, PER_QUESTION_COLUMNS)
    for column in PER_QUESTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = _coerce_metric_columns(frame)
    return frame


def load_chunking_ablation(path: Path = CHUNKING_ABLATION_PATH) -> pd.DataFrame:
    return _read_csv(
        path,
        ["chunk_size", "chunk_overlap", "file_count", "chunk_count", "avg_chunk_chars", "max_chunk_chars"],
    )


def load_embedding_comparison(path: Path = EMBEDDING_COMPARISON_PATH) -> pd.DataFrame:
    return _read_csv(
        path,
        ["model", "status", "sample_count", "embedding_dim", "embedding_ms", "error"],
    )


def best_strategy(summary: pd.DataFrame) -> str | None:
    if summary.empty or "strategy" not in summary.columns:
        return None

    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        strategy = row.get("strategy")
        metric_family, metric_columns = _metric_family_for_row(row)
        if not strategy or not metric_columns:
            continue
        numeric = pd.to_numeric(row[metric_columns], errors="coerce")
        mean_score = numeric.mean(skipna=True)
        if pd.isna(mean_score):
            continue
        rows.append({"strategy": strategy, "mean_score": float(mean_score), "metric_family": metric_family})
    scored = pd.DataFrame(rows)
    scored = scored.dropna(subset=["strategy", "mean_score"])
    if scored.empty:
        return None
    return str(scored.sort_values("mean_score", ascending=False).iloc[0]["strategy"])


def _finite_row_metrics(row: pd.Series, metrics: list[str]) -> list[str]:
    available = [metric for metric in metrics if metric in row.index]
    if not available:
        return []
    values = pd.to_numeric(row[available], errors="coerce")
    return [metric for metric in available if pd.notna(values.get(metric))]


def _metric_family_for_row(row: pd.Series) -> tuple[str | None, list[str]]:
    ragas_metrics = _finite_row_metrics(row, METRICS)
    if ragas_metrics:
        return "cloud_ragas", METRICS
    lexical_metrics = _finite_row_metrics(row, LEXICAL_METRICS)
    if lexical_metrics:
        return "offline_lexical", LEXICAL_METRICS
    return None, []


def _empty_metric_card_values(strategy: str | None = None) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "metric_family": None,
        "display_metrics": {metric: metric for metric in METRICS},
        **{metric: None for metric in METRICS},
    }


def metric_card_values(summary: pd.DataFrame, strategy: str | None = None) -> dict[str, Any]:
    selected_strategy = strategy or best_strategy(summary)
    if not selected_strategy:
        return _empty_metric_card_values()

    rows = summary[summary["strategy"] == selected_strategy]
    if rows.empty:
        return _empty_metric_card_values(selected_strategy)

    row = rows.iloc[0]
    metric_family, metric_columns = _metric_family_for_row(row)
    if not metric_family:
        return _empty_metric_card_values(selected_strategy)

    display_metrics = {metric: metric for metric in METRICS}
    if metric_family == "offline_lexical":
        display_metrics = dict(LEXICAL_TO_RAGAS_METRIC)

    values: dict[str, Any] = {
        "strategy": selected_strategy,
        "metric_family": metric_family,
        "display_metrics": display_metrics,
    }
    for metric in METRICS:
        source_metric = display_metrics[metric]
        value = row.get(source_metric)
        values[metric] = None if pd.isna(value) else float(value)
    return values


def filter_questions_below_threshold(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    available_metrics = [
        metric
        for metric in [*METRICS, *LEXICAL_METRICS]
        if metric in frame.columns and frame[metric].notna().any()
    ]
    if frame.empty or not available_metrics:
        return frame.copy()
    metric_values = frame[available_metrics].apply(pd.to_numeric, errors="coerce")
    mask = metric_values.lt(threshold).fillna(False).any(axis=1)
    return frame[mask].copy()


def eval_backend_counts(summary: pd.DataFrame, per_question: pd.DataFrame) -> dict[str, dict[str, int]]:
    summary_counts: dict[str, int] = {}
    question_counts: dict[str, int] = {}
    if "summary_backend" in summary.columns:
        summary_counts = {str(k): v for k, v in summary["summary_backend"].dropna().astype(str).value_counts().items()}
    if "evaluation_backend" in per_question.columns:
        question_counts = {str(k): v for k, v in per_question["evaluation_backend"].dropna().astype(str).value_counts().items()}
    return {"summary_backends": summary_counts, "question_backends": question_counts}


def cloud_ragas_status_message(summary: pd.DataFrame) -> str | None:
    if summary.empty or "cloud_status" not in summary.columns:
        return None
    statuses = summary["cloud_status"].fillna("").astype(str)
    attempted = statuses.isin({"succeeded", "degraded", "fallback_offline"})
    if not attempted.any():
        return None
    total = len(summary)
    succeeded = int(statuses.eq("succeeded").sum())
    degraded = int(statuses.eq("degraded").sum())
    fallback = int(statuses.eq("fallback_offline").sum())
    if degraded:
        return (
            f"Cloud RAGAS succeeded for {succeeded}/{total} strategies; "
            f"{degraded}/{total} returned partial metrics; "
            f"{fallback}/{total} fell back to offline."
        )
    return f"Cloud RAGAS succeeded for {succeeded}/{total} strategies; {fallback}/{total} fell back to offline."


def cloud_ragas_status_rows(summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["strategy", "summary_backend", "cloud_status", "cloud_error"]
    if summary.empty:
        return _empty_frame(columns)
    frame = summary.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    statuses = frame["cloud_status"].fillna("").astype(str)
    frame = frame[statuses.isin({"succeeded", "degraded", "fallback_offline"})].copy()
    frame["cloud_error"] = frame["cloud_error"].fillna("")
    return frame[columns]


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


def load_golden_questions(path: Path = GOLDEN_DATASET_PATH) -> pd.DataFrame:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return _empty_frame(["question", "source_doc", "source_slug"])
    if not isinstance(data, list):
        return _empty_frame(["question", "source_doc", "source_slug"])
    rows = [item for item in data if isinstance(item, dict)]
    frame = pd.DataFrame(rows)
    for column in ["question", "source_doc", "source_slug"]:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[["question", "source_doc", "source_slug"]]


def _ui_gate_overrides(st) -> dict[str, bool]:
    value = st.session_state.get(UI_GATE_OVERRIDES_KEY)
    return value if isinstance(value, dict) else {}


def _env_gate_default(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0") != "0"


def _gate_enabled(st, name: str, default: bool) -> bool:
    overrides = _ui_gate_overrides(st)
    if name in overrides:
        return bool(overrides[name])
    return _env_gate_default(name, default)


def _with_session_gate_env(st, names: list[str], fn):
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            default = bool(UI_TOGGLE_GATES[name]["default"])
            os.environ[name] = "1" if _gate_enabled(st, name, default=default) else "0"
        return fn()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


CURRENT_SOURCE_PATH = PROJECT_ROOT / "data" / "current_source.json"
PREPARED_SOURCE_PATH = PROJECT_ROOT / "data" / "prepared_source.json"


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clear_eval_artifacts() -> None:
    for path in [
        GOLDEN_DATASET_PATH,
        RAGAS_RESULTS_PATH,
        RAGAS_PER_QUESTION_PATH,
        CHUNKING_ABLATION_PATH,
        EMBEDDING_COMPARISON_PATH,
    ]:
        _remove_path(path)


def _clear_chroma_artifacts() -> None:
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    except Exception:
        _remove_path(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        return
    try:
        client.delete_collection("advanced_rag")
    except Exception as exc:
        not_found_error = getattr(getattr(chromadb, "errors", None), "NotFoundError", None)
        missing_collection = isinstance(not_found_error, type) and isinstance(exc, not_found_error)
        missing_collection = missing_collection or (
            isinstance(exc, ValueError) and "does not exist" in str(exc).lower()
        )
        if not missing_collection:
            raise


def clear_source_cache(clear_raw: bool = True) -> None:
    _clear_chroma_artifacts()
    for path in [CURRENT_SOURCE_PATH, PREPARED_SOURCE_PATH, QUERY_LOG_PATH, CHAT_HISTORY_DB_PATH]:
        _remove_path(path)
    clear_eval_artifacts()
    if clear_raw:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for path in RAW_DIR.iterdir():
            _remove_path(path)


def reset_session_source_state(st, reset_source_type: bool = False) -> None:
    st.session_state.pop(PREPARED_SOURCE_KEY, None)
    st.session_state.pop("prepared_files", None)
    st.session_state.pop(CHAT_MESSAGES_KEY, None)
    st.session_state.pop(ACTIVE_CHAT_SOURCE_KEY, None)
    if reset_source_type:
        st.session_state[SOURCE_TYPE_KEY] = DEFAULT_SOURCE_TYPE


def load_current_source(path: Path = CURRENT_SOURCE_PATH) -> dict[str, Any] | None:
    """Load data/current_source.json; return None if missing."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_prepared_source(path: Path | None = None) -> dict[str, Any] | None:
    target = path or PREPARED_SOURCE_PATH
    if not target.exists():
        return None
    try:
        prepared = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return prepared if isinstance(prepared, dict) else None


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


def _chat_db_connection(path: Path = CHAT_HISTORY_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_slug TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_messages_source_id ON chat_messages(source_slug, id)"
    )
    return connection


def load_persisted_chat_messages(
    source_slug: str,
    db_path: Path = CHAT_HISTORY_DB_PATH,
    limit: int = MAX_PERSISTED_CHAT_MESSAGES,
) -> list[dict[str, str]]:
    """Load persisted display history for a single indexed source."""

    if not source_slug or not db_path.exists():
        return []
    connection = _chat_db_connection(db_path)
    try:
        rows = connection.execute(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM chat_messages
                WHERE source_slug = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (source_slug, limit),
        ).fetchall()
    finally:
        connection.close()
    return [{"role": str(role), "content": str(content)} for role, content in rows]


def append_persisted_chat_message(
    source_slug: str,
    message: dict[str, Any],
    db_path: Path = CHAT_HISTORY_DB_PATH,
) -> None:
    role = str(message.get("role", ""))
    content = str(message.get("content", ""))
    if not source_slug or role not in {"user", "assistant"} or not content:
        return
    connection = _chat_db_connection(db_path)
    try:
        connection.execute(
            """
            INSERT INTO chat_messages(source_slug, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (source_slug, role, content, pd.Timestamp.utcnow().isoformat()),
        )
        connection.commit()
    finally:
        connection.close()


def clear_persisted_chat_history(source_slug: str | None = None, db_path: Path = CHAT_HISTORY_DB_PATH) -> None:
    if not db_path.exists():
        return
    connection = _chat_db_connection(db_path)
    try:
        if source_slug:
            connection.execute("DELETE FROM chat_messages WHERE source_slug = ?", (source_slug,))
        else:
            connection.execute("DELETE FROM chat_messages")
        connection.commit()
    finally:
        connection.close()


def _has_raw_source_files(raw_dir: Path = RAW_DIR) -> bool:
    """Return True if data/raw/ has any files (recursively)."""
    if not raw_dir.exists():
        return False
    return any(raw_dir.rglob("*"))


def _has_chroma_index(chroma_dir: Path | None = None) -> bool:
    target = chroma_dir or CHROMA_DIR
    if not isinstance(target, Path):
        return bool(target.exists())
    return target.exists() and any(target.rglob("*"))


def get_source_badge_state(
    current_source_path: Path = CURRENT_SOURCE_PATH,
    chroma_dir: Path = CHROMA_DIR,
    raw_dir: Path = RAW_DIR,
) -> str:
    """Return 'green' if indexed, 'yellow' if prepared, 'grey' otherwise."""
    has_source = current_source_path.exists()
    has_chroma = _has_chroma_index(chroma_dir)
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
    _index, nodes = build_index(source_files=files, raw_dir=RAW_DIR)
    return nodes


def generate_pre_questions_inline() -> list[dict[str, str]]:
    from evaluate import generate_pre_questions

    return generate_pre_questions()


def run_evaluation_inline(use_real_ragas: bool = True) -> None:
    """Run full evaluation pipeline inline (blocking). Calls evaluate.main()."""
    from evaluate import main as eval_main

    eval_main(use_real_ragas=use_real_ragas)


def run_chunking_ablation_inline() -> list[dict[str, Any]]:
    from ingestion import run_chunking_ablation

    return run_chunking_ablation(raw_dir=RAW_DIR, output_path=CHUNKING_ABLATION_PATH)


def run_embedding_comparison_inline() -> list[dict[str, Any]]:
    from ingestion import run_embedding_model_comparison

    return run_embedding_model_comparison(
        raw_dir=RAW_DIR,
        allow_model_downloads=_env_gate_default("ALLOW_MODEL_DOWNLOADS", default=False),
        output_path=EMBEDDING_COMPARISON_PATH,
    )


def _run_with_timeout(fn, timeout_seconds: float) -> bool:
    import concurrent.futures as _cf

    if timeout_seconds <= 0:
        return False

    _pool = _cf.ThreadPoolExecutor(max_workers=1)
    _future = _pool.submit(fn)
    try:
        _future.result(timeout=timeout_seconds)
        return True
    except _cf.TimeoutError:
        _future.cancel()
        return False
    finally:
        _pool.shutdown(wait=False, cancel_futures=True)


def last_eval_date(path: Path = RAGAS_RESULTS_PATH) -> str:
    if not path.exists():
        return "Not available"
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")


def build_grouped_bar_chart(summary: pd.DataFrame):
    import matplotlib.pyplot as plt

    if summary.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.text(0.5, 0.5, "No evaluation data", ha="center", va="center")
        ax.axis("off")
        return fig

    ragas_metrics = [metric for metric in METRICS if metric in summary.columns and pd.to_numeric(summary[metric], errors="coerce").notna().any()]
    lexical_metrics = [metric for metric in LEXICAL_METRICS if metric in summary.columns and pd.to_numeric(summary[metric], errors="coerce").notna().any()]
    metric_groups = [
        ("Cloud RAGAS metrics", ragas_metrics, _rows_with_any_metric(summary, ragas_metrics)),
        ("Offline lexical metrics", lexical_metrics, _rows_with_any_metric(summary, lexical_metrics)),
    ]
    metric_groups = [(title, metrics, rows) for title, metrics, rows in metric_groups if metrics and not rows.empty]
    if not metric_groups:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.text(0.5, 0.5, "No finite evaluation metrics", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(len(metric_groups), 1, figsize=(9, 4.5 * len(metric_groups)))
    if len(metric_groups) == 1:
        axes = [axes]
    for ax, (title, metrics, rows) in zip(axes, metric_groups):
        indexed = rows.set_index("strategy")
        indexed[metrics].plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Score")
        lower, upper = _metric_axis_bounds(indexed[metrics])
        ax.set_ylim(lower, upper)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _rows_with_any_metric(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if not metrics:
        return summary.iloc[0:0].copy()
    metric_values = summary[metrics].apply(pd.to_numeric, errors="coerce")
    return summary[metric_values.notna().any(axis=1)].copy()


def _metric_axis_bounds(values: pd.DataFrame) -> tuple[float, float]:
    numeric = values.apply(pd.to_numeric, errors="coerce").stack().dropna()
    if numeric.empty:
        return 0.0, 1.0
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    lower = min(0.0, minimum)
    upper = max(1.0, maximum)
    if lower < 0:
        lower -= max(0.05, abs(lower) * 0.1)
    if upper > 1:
        upper += max(0.05, abs(upper) * 0.1)
    return lower, upper


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def run_query(query: str, strategy: str) -> dict[str, Any]:
    from pipeline import answer_query

    return answer_query(query, strategy=strategy, allow_index_build=False)


def run_chat_query(message: str, history: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    from pipeline import chat_query

    return chat_query(message, history=history, strategy=strategy, allow_index_build=False)


def prepare_sources_for_app(
    sources: list[str],
    allow_github_fetch: bool = False,
    allow_huggingface_fetch: bool = False,
    clear_existing: bool = False,
) -> list[Path]:
    from source_loader import prepare_sources

    cleaned = [source.strip() for source in sources if source.strip()]
    if not cleaned:
        return []
    return prepare_sources(
        cast(list[str | Path], cleaned),
        raw_dir=RAW_DIR,
        allow_github_fetch=allow_github_fetch,
        allow_huggingface_fetch=allow_huggingface_fetch,
        clear_existing=clear_existing,
    )


def prepare_uploaded_files_for_app(uploaded_files: list[Any], clear_existing: bool = False) -> list[Path]:
    from source_loader import prepare_uploaded_files

    if not uploaded_files:
        return []
    return prepare_uploaded_files(
        uploaded_files,
        raw_dir=RAW_DIR,
        clear_existing=clear_existing,
    )


def _prepared_source_state(st) -> dict[str, Any] | None:
    value = st.session_state.get(PREPARED_SOURCE_KEY)
    return value if isinstance(value, dict) else None


def _source_slug_matches(left: Any, right: Any) -> bool:
    return str(left or "").lower() == str(right or "").lower()


def _current_source_for_ui(st) -> dict[str, Any] | None:
    prepared = _prepared_source_state(st) or load_prepared_source()
    indexed = load_current_source()
    if prepared and not _source_slug_matches(prepared.get("source_slug"), (indexed or {}).get("source_slug")):
        return prepared
    return indexed


def _source_is_indexed(current_source: dict[str, Any] | None) -> bool:
    if current_source is None or not _has_chroma_index():
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
    return indexed is not None and _has_chroma_index()


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
    prepared_source = {
        "source_slug": _prepared_source_slug(prepared),
        "source_input": source_input,
        "source_type": source_type,
        "indexed_at": None,
        "file_count": len(prepared),
    }
    st.session_state["prepared_files"] = prepared
    st.session_state[PREPARED_SOURCE_KEY] = prepared_source
    PREPARED_SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPARED_SOURCE_PATH.write_text(json.dumps(prepared_source, indent=2), encoding="utf-8")


def _clear_prepared_source(st) -> None:
    st.session_state.pop(PREPARED_SOURCE_KEY, None)
    st.session_state.pop("prepared_files", None)
    _remove_path(PREPARED_SOURCE_PATH)


def _prepared_source_type(source_type: str) -> str:
    if source_type == "HuggingFace model/dataset":
        return "huggingface"
    if source_type == "GitHub repo":
        return "github"
    if source_type == "Upload files":
        return "upload"
    return "local"


def _render_prepared_files(st, prepared: list[Path]) -> None:
    st.dataframe(
        pd.DataFrame({"file": [path.as_posix() for path in prepared]}),
        use_container_width=True,
    )


def _prepared_source_slug(prepared: list[Path]) -> str:
    if not prepared:
        return "unknown"
    try:
        relative = prepared[0].resolve(strict=False).relative_to(RAW_DIR.resolve(strict=False))
        if relative.parts:
            return relative.parts[0]
    except ValueError:
        pass
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


def _allow_index_build_enabled(st) -> bool:
    return _gate_enabled(st, "ALLOW_INDEX_BUILD", default=True)


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
    return bool(prepared and indexed and _source_slug_matches(prepared.get("source_slug"), indexed.get("source_slug")))


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


def _render_gate_toggles(st) -> None:
    st.sidebar.header("Session gates")
    overrides = dict(_ui_gate_overrides(st))
    for name, config in UI_TOGGLE_GATES.items():
        default_value = _env_gate_default(name, config["default"])
        value = st.sidebar.checkbox(
            config["label"],
            value=bool(overrides.get(name, default_value)),
            help=f"Default from .env: {'on' if default_value else 'off'}. Override applies only to this Streamlit session.",
        )
        overrides[name] = value
    st.session_state[UI_GATE_OVERRIDES_KEY] = overrides


def _sidebar_badge_state_for_source(current_source: dict[str, Any] | None) -> str:
    if current_source is None:
        return "grey"
    return "green" if _source_is_indexed(current_source) else "yellow"


def _render_sidebar(st) -> None:
    current_source = _source_ui_state(st)
    badge_state = _sidebar_badge_state_for_source(current_source) if current_source is not None else get_source_badge_state()
    _render_source_badge(st, badge_state, current_source)
    _render_gate_toggles(st)

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


def _render_source_status_panel(st) -> None:
    current_source = _source_ui_state(st)
    badge_state = _sidebar_badge_state_for_source(current_source) if current_source is not None else get_source_badge_state()
    if badge_state == "green":
        slug = current_source.get("source_slug", "unknown") if current_source else "unknown"
        indexed_at = current_source.get("indexed_at", "unknown") if current_source else "unknown"
        st.success(f"Source indexed: {slug}")
        st.caption(f"Indexed at: {indexed_at}")
    elif badge_state == "yellow":
        slug = current_source.get("source_slug", "unknown") if current_source else "unknown"
        st.warning(f"Source prepared but not indexed: {slug}")
    else:
        st.error("No source prepared")


def _render_gate_toggles_panel(st) -> None:
    st.subheader("Session gates")
    overrides = dict(_ui_gate_overrides(st))
    for name, config in UI_TOGGLE_GATES.items():
        default_value = _env_gate_default(name, config["default"])
        value = st.checkbox(
            config["label"],
            value=bool(overrides.get(name, default_value)),
            help=f"Default from .env: {'on' if default_value else 'off'}. Override applies only to this Streamlit session.",
        )
        overrides[name] = value
    st.session_state[UI_GATE_OVERRIDES_KEY] = overrides


def _render_run_metadata_panel(st) -> None:
    stats = dataset_stats()
    with st.expander("Run metadata", expanded=False):
        st.write("Model")
        st.json(MODEL_INFO)
        st.write("Dataset stats")
        st.json(stats)
        st.write(f"Last eval: {last_eval_date()}")

        data = RAGAS_RESULTS_PATH.read_bytes() if RAGAS_RESULTS_PATH.exists() else b""
        st.download_button(
            "Download results CSV",
            data=data,
            file_name="ragas_results.csv",
            mime="text/csv",
            disabled=not bool(data),
        )


def _render_source_workspace(st) -> None:
    _render_source_status_panel(st)
    _render_sources_tab(st)
    _render_gate_toggles_panel(st)
    _render_run_metadata_panel(st)


def _render_sources_tab(st) -> None:
    st.subheader("Source preparation")

    if st.button("Clear active source cache"):
        clear_source_cache()
        reset_session_source_state(st, reset_source_type=True)
        st.success("Active source cache cleared.")
        st.rerun()

    source_type = st.radio(
        "Source type",
        [DEFAULT_SOURCE_TYPE, "Local directory", "GitHub repo", "HuggingFace model/dataset"],
        key=SOURCE_TYPE_KEY,
    )

    # Adaptive input field
    uploaded_files: list[Any] = []
    source_text = ""
    if source_type == DEFAULT_SOURCE_TYPE:
        from source_loader import SOURCE_EXTENSIONS

        allowed_types = sorted(ext.lstrip(".") for ext in SOURCE_EXTENSIONS)
        uploaded_files = st.file_uploader(
            "Upload source files",
            type=allowed_types,
            accept_multiple_files=True,
            help="Use this for public deployments. Browser uploads are copied into the app session before indexing.",
        )
    elif source_type == "Local directory":
        st.caption("Local paths only work when Streamlit runs on the same machine as the files. Use uploads for public deployments.")
        source_text = st.text_input("Local path", placeholder="C:\\path\\to\\project")
    elif source_type == "GitHub repo":
        source_text = st.text_input("GitHub URL", placeholder="https://github.com/user/repo")
    else:
        source_text = st.text_input("HuggingFace URL", placeholder="hf:owner/model or https://huggingface.co/owner/model")

    allow_hf_fetch = _gate_enabled(st, "ALLOW_HF_FETCH", default=False)
    allow_github_fetch = _gate_enabled(st, "ALLOW_GITHUB_FETCH", default=False)

    # Prepare sources
    if st.button("Prepare sources"):
        if source_type == DEFAULT_SOURCE_TYPE and not uploaded_files:
            st.warning("No uploaded files were provided.")
        elif source_type != DEFAULT_SOURCE_TYPE and (not source_text or not source_text.strip()):
            st.warning("No sources were provided.")
        else:
            sources = [source_text.strip()] if source_text else []
            try:
                if source_type == DEFAULT_SOURCE_TYPE:
                    prepared = prepare_uploaded_files_for_app(
                        uploaded_files,
                        clear_existing=True,
                    )
                elif source_type == "HuggingFace model/dataset":
                    prepared = prepare_sources_for_app(
                        sources,
                        allow_github_fetch=False,
                        allow_huggingface_fetch=allow_hf_fetch,
                        clear_existing=True,
                    )
                elif source_type == "GitHub repo":
                    prepared = prepare_sources_for_app(
                        sources,
                        allow_github_fetch=allow_github_fetch,
                        clear_existing=True,
                    )
                else:
                    prepared = prepare_sources_for_app(
                        sources,
                        allow_github_fetch=False,
                        allow_huggingface_fetch=False,
                        clear_existing=True,
                    )
            except Exception as exc:
                st.error(str(exc))
                return
            if prepared:
                clear_source_cache(clear_raw=False)
                reset_session_source_state(st)
                st.success(f"Prepared {len(prepared)} files under data/raw.")
                source_input = sources[0] if sources else f"uploaded:{', '.join(str(getattr(file, 'name', 'file')) for file in uploaded_files)}"
                _store_prepared_source(st, prepared, source_input, _prepared_source_type(source_type))
                st.dataframe(
                    pd.DataFrame({"file": [path.as_posix() for path in prepared]}),
                    use_container_width=True,
                )
                st.info(f"Prepared source: {_prepared_source_slug(prepared)}")
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
            elif not _allow_index_build_enabled(st):
                _show_allow_index_build_error(st)
            else:
                progress = st.progress(0.0, "Building index...")
                try:
                    nodes = run_build_index(st.session_state.get("prepared_files"))
                    progress.progress(1.0)
                    current = load_current_source()
                    slug = current.get("source_slug", "unknown") if current else "unknown"
                    _clear_prepared_source(st)
                    st.session_state.pop(CHAT_MESSAGES_KEY, None)
                    st.session_state.pop(ACTIVE_CHAT_SOURCE_KEY, None)
                    clear_eval_artifacts()
                    st.success(f"Index built successfully. {len(nodes)} chunks for source '{slug}'.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Index build failed: {exc}")


def _sync_chat_history_to_source(st, current_source: dict[str, Any] | None) -> None:
    source_slug = current_source.get("source_slug") if current_source else None
    if not source_slug:
        st.session_state.pop(ACTIVE_CHAT_SOURCE_KEY, None)
        st.session_state.pop(CHAT_MESSAGES_KEY, None)
        return
    previous_slug = st.session_state.get(ACTIVE_CHAT_SOURCE_KEY)
    st.session_state[ACTIVE_CHAT_SOURCE_KEY] = source_slug
    if previous_slug is not None and previous_slug != source_slug:
        st.session_state[CHAT_MESSAGES_KEY] = load_persisted_chat_messages(str(source_slug))
    elif CHAT_MESSAGES_KEY not in st.session_state:
        st.session_state[CHAT_MESSAGES_KEY] = load_persisted_chat_messages(str(source_slug))


def _chat_history(st) -> list[dict[str, Any]]:
    if CHAT_MESSAGES_KEY not in st.session_state:
        st.session_state[CHAT_MESSAGES_KEY] = []
    return st.session_state[CHAT_MESSAGES_KEY]


def _retrieval_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": str(message.get("content", ""))}
        for message in messages
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]


def _citation_text(citation: dict[str, Any], index: int) -> str:
    source = citation.get("source_doc") or citation.get("source") or "unknown source"
    return f"{index}. {source}"


def _render_citations(st, citations: list[dict[str, Any]]) -> None:
    if not citations:
        st.caption("No citations returned.")
        return
    st.write("Sources")
    for index, citation in enumerate(citations[:5], start=1):
        st.caption(_citation_text(citation, index))


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


def _stream_text(text: str):
    words = text.split(" ")
    if not words:
        yield ""
        return
    for index, word in enumerate(words):
        yield word if index == 0 else f" {word}"


def _render_chat_message(st, message: dict[str, Any], stream_content: bool = False) -> None:
    with st.chat_message(message.get("role", "assistant")):
        role = message.get("role", "assistant")
        content = message.get("content", "")
        if role == "assistant" and stream_content and hasattr(st, "write_stream"):
            st.write_stream(_stream_text(str(content)))
        else:
            st.write(content)
        if role == "assistant":
            if "citations" in message:
                _render_citations(st, message.get("citations", []))
            if "trace" in message:
                _render_trace_debug(st, message.get("trace"))


def _streamlit_container(st, key: str):
    if hasattr(st, "container"):
        return st.container(key=key)
    return nullcontext()


def _render_query_tab(st) -> None:
    current_source = _source_ui_state(st)

    if not _query_should_render_controls(st):
        _render_query_blocked(st)
        return

    _render_query_ready(st)
    _sync_chat_history_to_source(st, current_source)

    strategy = st.selectbox("Strategy", STRATEGIES, index=STRATEGIES.index("hybrid_rerank"))

    messages = _chat_history(st)
    history_panel = _streamlit_container(st, "chat_history_panel")
    with history_panel:
        for message in messages:
            _render_chat_message(st, message)

    with _streamlit_container(st, "chat_composer_panel"):
        prompt = st.chat_input("Ask about the indexed project context")
    if not prompt or not prompt.strip():
        return

    source_slug = str(current_source.get("source_slug", "")) if current_source else ""
    user_message = {"role": "user", "content": prompt.strip()}
    history = _retrieval_history(messages)
    messages.append(user_message)
    append_persisted_chat_message(source_slug, user_message)
    with history_panel:
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
    append_persisted_chat_message(source_slug, assistant_message)
    with history_panel:
        _render_chat_message(st, assistant_message, stream_content=True)


def _cloud_ragas_provider_names() -> list[str]:
    import cloud_ragas

    return [provider.name for provider in cloud_ragas.providers_from_env()]


def _show_cloud_ragas_provider_missing(st) -> None:
    st.info(
        "Cloud RAGAS needs at least one configured provider key in the hosted environment: "
        "GEMINI_API_KEY, GROQ_API_KEY, or GITHUB_MODELS_TOKEN plus GITHUB_MODELS_MODEL. "
        "Fast offline evaluation remains available without provider keys."
    )


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

    if st.button("Generate pre-questions"):
        if not _eval_should_allow_run(st):
            _render_eval_run_blocked(st)
        else:
            with st.spinner("Generating source-specific pre-questions..."):
                try:
                    questions = _with_session_gate_env(
                        st,
                        ["USE_CLOUD_FREE_TIER_RAGAS", "ALLOW_CLOUD_FREE_TIER", "ALLOW_INDEX_BUILD"],
                        generate_pre_questions_inline,
                    )
                    st.success(f"Generated {len(questions)} source-specific pre-questions.")
                    st.rerun()
                except Exception as exc:
                    _show_eval_failure(st, exc)

    if st.button("Run fast evaluation"):
        if not _eval_should_allow_run(st):
            _render_eval_run_blocked(st)
        else:
            with st.spinner("Running fast offline evaluation..."):
                try:
                    previous_cloud_chat = os.environ.get("ALLOW_CLOUD_CHAT")
                    try:
                        os.environ["ALLOW_CLOUD_CHAT"] = "0"
                        _with_session_gate_env(
                            st,
                            ["ALLOW_INDEX_BUILD"],
                            lambda: run_evaluation_inline(use_real_ragas=False),
                        )
                    finally:
                        if previous_cloud_chat is None:
                            os.environ.pop("ALLOW_CLOUD_CHAT", None)
                        else:
                            os.environ["ALLOW_CLOUD_CHAT"] = previous_cloud_chat
                    _show_eval_success(st)
                    st.rerun()
                except Exception as exc:
                    _show_eval_failure(st, exc)

    if st.button("Run Cloud RAGAS"):
        if not _eval_should_allow_run(st):
            _render_eval_run_blocked(st)
        elif not _cloud_ragas_provider_names():
            _show_cloud_ragas_provider_missing(st)
        else:
            cloud_ragas_timeout_seconds = float(os.getenv("CLOUD_RAGAS_TIMEOUT_SECONDS", "300"))
            spinner_msg = (
                f"Running Cloud RAGAS evaluation "
                f"(timeout: {cloud_ragas_timeout_seconds:g}s — "
                "check terminal for live provider logs)..."
            )
            with st.spinner(spinner_msg):
                def _run_cloud_ragas():
                    previous = {name: os.environ.get(name) for name in ["USE_CLOUD_FREE_TIER_RAGAS", "ALLOW_CLOUD_FREE_TIER", "ALLOW_CLOUD_CHAT"]}
                    try:
                        os.environ["USE_CLOUD_FREE_TIER_RAGAS"] = "1"
                        os.environ["ALLOW_CLOUD_FREE_TIER"] = "1"
                        os.environ["ALLOW_CLOUD_CHAT"] = "0"
                        return _with_session_gate_env(
                            st,
                            ["ALLOW_INDEX_BUILD"],
                            lambda: run_evaluation_inline(use_real_ragas=True),
                        )
                    finally:
                        for name, value in previous.items():
                            if value is None:
                                os.environ.pop(name, None)
                            else:
                                os.environ[name] = value

                try:
                    completed = _run_with_timeout(_run_cloud_ragas, cloud_ragas_timeout_seconds)
                    if completed:
                        _show_eval_success(st)
                        st.rerun()
                    else:
                        st.error(
                            f"Cloud RAGAS timed out after {cloud_ragas_timeout_seconds:g}s. "
                            "Likely cause: all providers are rate-limited (429). "
                            "Wait 60s and retry, or increase MAX_CLOUD_CALLS / CLOUD_RAGAS_TIMEOUT_SECONDS in .env."
                        )
                except Exception as exc:
                    _show_eval_failure(st, exc)

    if st.button("Run chunking ablation"):
        if not _eval_should_allow_run(st):
            _render_eval_run_blocked(st)
        else:
            with st.spinner("Running chunk-size ablation..."):
                try:
                    rows = _with_session_gate_env(
                        st,
                        ["ALLOW_DOCS_DOWNLOAD"],
                        run_chunking_ablation_inline,
                    )
                    st.success(f"Chunking ablation completed for {len(rows)} chunk sizes.")
                    st.rerun()
                except Exception as exc:
                    _show_eval_failure(st, exc)

    if st.button("Run embedding comparison"):
        if not _eval_should_allow_run(st):
            _render_eval_run_blocked(st)
        else:
            with st.spinner("Running embedding model comparison..."):
                try:
                    rows = _with_session_gate_env(
                        st,
                        ["ALLOW_DOCS_DOWNLOAD", "ALLOW_MODEL_DOWNLOADS"],
                        run_embedding_comparison_inline,
                    )
                    st.success(f"Embedding comparison completed for {len(rows)} models.")
                    st.rerun()
                except Exception as exc:
                    _show_eval_failure(st, exc)

    st.subheader("Evaluation summary")
    display_metrics = cards.get("display_metrics", {metric: metric for metric in METRICS})
    for metric_row in [METRICS[:2], METRICS[2:]]:
        card_columns = st.columns(2)
        for column, metric in zip(card_columns, metric_row, strict=True):
            column.metric(display_metrics.get(metric, metric), _format_metric(cards[metric]))

    best = cards.get("strategy")
    if best:
        metric_family = cards.get("metric_family")
        suffix = f" ({metric_family})" if metric_family else ""
        st.success(f"Best strategy: {best}{suffix}")
    else:
        st.warning("No strategy scores available.")

    st.write("Evaluation backend")
    st.json(backends)

    cloud_status_message = cloud_ragas_status_message(summary)
    if cloud_status_message:
        st.info(cloud_status_message)
        status_rows = cloud_ragas_status_rows(summary)
        if not status_rows.empty:
            st.dataframe(status_rows, use_container_width=True)

    st.pyplot(build_grouped_bar_chart(summary))

    st.subheader("Per-question scores")
    threshold = st.slider("Show rows with any metric below", 0.0, 1.0, 0.5, 0.05)
    filtered = filter_questions_below_threshold(per_question, threshold)
    st.dataframe(filtered, use_container_width=True)

    with st.expander("Generated pre-questions", expanded=False):
        st.subheader("Generated pre-questions")
        st.dataframe(load_golden_questions(), use_container_width=True)

    with st.expander("Chunking ablation", expanded=False):
        st.subheader("Chunking ablation")
        st.dataframe(load_chunking_ablation(), use_container_width=True)

    with st.expander("Embedding comparison", expanded=False):
        st.subheader("Embedding comparison")
        st.dataframe(load_embedding_comparison(), use_container_width=True)


def render_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Advanced RAG", layout="wide", initial_sidebar_state="collapsed")
    _inject_app_chrome(st)
    _render_workbench_header(st)

    source_col, main_col = st.columns([0.78, 2.2], gap="medium")
    with source_col:
        with st.container(key="source_index_panel"):
            _render_source_workspace(st)
    with main_col:
        with st.container(key="main_workspace_panel"):
            with st.container(key="workspace_page_nav"):
                page = st.radio("Workspace page", ["Chat", "RAGAS evaluation"], key=WORKSPACE_PAGE_KEY)
            if page == "RAGAS evaluation":
                with st.container(key="eval_page_content"):
                    st.subheader("RAGAS evaluation")
                    _render_eval_tab(st)
            else:
                with st.container(key="chat_page_content"):
                    st.subheader("Chat")
                    _render_query_tab(st)


if __name__ == "__main__":
    render_app()
