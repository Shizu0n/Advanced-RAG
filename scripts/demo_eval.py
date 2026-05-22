from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate
from ingestion import build_index
from pipeline import chat_query
from source_loader import prepare_sources

HF_SOURCE = "hf:Shizu0n/phi3-mini-sql-generator"
DEMO_QUESTION = "qual a dataset usada no fine tunning desse model do hugging face?"
PROVIDER_KEY_NAMES = ("GEMINI_API_KEY", "GROQ_API_KEY", "GITHUB_MODELS_TOKEN")
CLOUD_EVAL_GATES = ("USE_GEMINI_FREE_RAGAS", "ALLOW_CLOUD_FREE_TIER")
CLOUD_RAGAS_KEY = "GEMINI_API_KEY"
EXPECTED_STRATEGIES = {"semantic_only", "bm25_only", "hybrid_no_rerank", "hybrid_rerank"}


def _has_cloud_ragas_key() -> bool:
    return bool(os.getenv(CLOUD_RAGAS_KEY))


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    details: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "details": self.details}


def _has_provider_key() -> bool:
    return any(bool(os.getenv(name)) for name in PROVIDER_KEY_NAMES)


def cloud_eval_status() -> dict[str, str]:
    missing: list[str] = []
    if os.getenv("USE_GEMINI_FREE_RAGAS") != "1":
        missing.append("USE_GEMINI_FREE_RAGAS=1")
    if os.getenv("ALLOW_CLOUD_FREE_TIER") != "1":
        missing.append("ALLOW_CLOUD_FREE_TIER=1")
    if not _has_cloud_ragas_key():
        missing.append("GEMINI_API_KEY for cloud RAGAS")
    if missing:
        return {"status": "skipped", "reason": "Missing " + ", ".join(missing)}
    return {"status": "ready", "reason": "Cloud RAGAS prerequisites are configured."}


def run_stage(name: str, action: Callable[[], str]) -> StageResult:
    details = action()
    return StageResult(name=name, status="completed", details=details)


def run_prepare() -> str:
    if os.getenv("ALLOW_HF_FETCH") != "1":
        raise RuntimeError("Set ALLOW_HF_FETCH=1 to allow the HuggingFace demo source fetch.")
    copied = prepare_sources([HF_SOURCE])
    return f"prepared {len(copied)} file(s) from {HF_SOURCE}"


def run_index() -> str:
    _, nodes = build_index()
    return f"indexed {len(nodes)} chunk(s)"


def run_query() -> str:
    previous_cloud_chat = os.environ.get("ALLOW_CLOUD_CHAT")
    os.environ["ALLOW_CLOUD_CHAT"] = "0"
    try:
        result = chat_query(DEMO_QUESTION, strategy="hybrid_rerank", allow_index_build=False)
    finally:
        if previous_cloud_chat is None:
            os.environ.pop("ALLOW_CLOUD_CHAT", None)
        else:
            os.environ["ALLOW_CLOUD_CHAT"] = previous_cloud_chat
    synthesis = result.get("trace", {}).get("synthesis", {})
    return f"answer={result.get('answer', '')}\nsynthesis={synthesis}"


def run_offline_eval() -> str:
    previous_values = {name: os.environ.pop(name, None) for name in CLOUD_EVAL_GATES}
    try:
        evaluate.main()
    finally:
        for name, value in previous_values.items():
            if value is not None:
                os.environ[name] = value
    return "offline evaluation completed"


def _cloud_eval_result_status(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError):
        return {
            "status": "skipped",
            "reason": "Cloud RAGAS prerequisites were configured, but evaluation fell back to offline heuristics.",
        }
    if not rows or not reader.fieldnames or "summary_backend" not in reader.fieldnames:
        return {
            "status": "skipped",
            "reason": "Cloud RAGAS prerequisites were configured, but evaluation fell back to offline heuristics.",
        }
    strategies = {row.get("strategy") for row in rows}
    if strategies != EXPECTED_STRATEGIES:
        return {
            "status": "skipped",
            "reason": "Cloud RAGAS did not produce all strategy summaries.",
        }
    summary_backends = [(row.get("summary_backend") or "").strip() for row in rows]
    if summary_backends and all(backend == "gemini_free_tier_ragas" for backend in summary_backends):
        return {"status": "completed", "reason": "Cloud RAGAS evaluation completed."}
    return {
        "status": "skipped",
        "reason": "Cloud RAGAS prerequisites were configured, but evaluation fell back to offline heuristics.",
    }


def run_cloud_eval_if_available() -> dict[str, str]:
    status = cloud_eval_status()
    if status["status"] != "ready":
        return status
    evaluate.main()
    return _cloud_eval_result_status(evaluate.RAGAS_RESULTS_PATH)


def run_cloud_eval_stage() -> StageResult:
    status = run_cloud_eval_if_available()
    return StageResult(
        name="cloud_eval",
        status=status["status"],
        details=status["reason"],
    )


def run_demo() -> list[StageResult]:
    return [
        run_stage("prepare", run_prepare),
        run_stage("index", run_index),
        run_stage("query", run_query),
        run_stage("offline_eval", run_offline_eval),
        run_cloud_eval_stage(),
    ]


def main() -> None:
    for result in run_demo():
        print(f"[{result.status}] {result.name}: {result.details}")


if __name__ == "__main__":
    main()
