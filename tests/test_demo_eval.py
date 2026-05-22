import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import demo_eval


class DemoEvalRunnerTests(unittest.TestCase):
    def test_cloud_eval_is_skipped_without_gates_and_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            status = demo_eval.cloud_eval_status()

        self.assertEqual(status["status"], "skipped")
        self.assertIn("USE_GEMINI_FREE_RAGAS=1", status["reason"])
        self.assertIn("ALLOW_CLOUD_FREE_TIER=1", status["reason"])
        self.assertIn("GEMINI_API_KEY for cloud RAGAS", status["reason"])

    def test_cloud_eval_is_skipped_without_gemini_key_even_if_other_provider_key_exists(self):
        fake_evaluate = Mock()
        with (
            patch.dict(
                os.environ,
                {
                    "USE_GEMINI_FREE_RAGAS": "1",
                    "ALLOW_CLOUD_FREE_TIER": "1",
                    "GROQ_API_KEY": "test-groq-key",
                },
                clear=True,
            ),
            patch.object(demo_eval, "evaluate", fake_evaluate),
        ):
            status = demo_eval.run_cloud_eval_if_available()

        fake_evaluate.main.assert_not_called()
        self.assertEqual(status["status"], "skipped")
        self.assertIn("GEMINI_API_KEY", status["reason"])

    def test_cloud_eval_reports_completed_when_all_strategy_results_show_gemini_backend(self):
        fake_evaluate = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "ragas_results.csv"
            rows = "".join(
                f"{strategy},1.0,1.0,1.0,1.0,gemini_free_tier_ragas\n"
                for strategy in demo_eval.EXPECTED_STRATEGIES
            )
            results_path.write_text(
                "strategy,faithfulness,answer_relevancy,context_recall,context_precision,summary_backend\n"
                + rows,
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "USE_GEMINI_FREE_RAGAS": "1",
                        "ALLOW_CLOUD_FREE_TIER": "1",
                        "GEMINI_API_KEY": "test-key",
                    },
                    clear=True,
                ),
                patch.object(demo_eval, "evaluate", fake_evaluate),
                patch.object(demo_eval.evaluate, "RAGAS_RESULTS_PATH", results_path),
            ):
                status = demo_eval.run_cloud_eval_if_available()

        fake_evaluate.main.assert_called_once_with()
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["reason"], "Cloud RAGAS evaluation completed.")

    def test_cloud_eval_reports_skipped_when_strategy_results_are_partial(self):
        fake_evaluate = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "ragas_results.csv"
            results_path.write_text(
                "strategy,faithfulness,answer_relevancy,context_recall,context_precision,summary_backend\n"
                "hybrid_rerank,1.0,1.0,1.0,1.0,gemini_free_tier_ragas\n",
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "USE_GEMINI_FREE_RAGAS": "1",
                        "ALLOW_CLOUD_FREE_TIER": "1",
                        "GEMINI_API_KEY": "test-key",
                    },
                    clear=True,
                ),
                patch.object(demo_eval, "evaluate", fake_evaluate),
                patch.object(demo_eval.evaluate, "RAGAS_RESULTS_PATH", results_path),
            ):
                status = demo_eval.run_cloud_eval_if_available()

        fake_evaluate.main.assert_called_once_with()
        self.assertEqual(status["status"], "skipped")
        self.assertIn("all strategy", status["reason"])

    def test_cloud_eval_reports_skipped_when_results_include_non_cloud_rows(self):
        fake_evaluate = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "ragas_results.csv"
            results_path.write_text(
                "strategy,faithfulness,answer_relevancy,context_recall,context_precision,summary_backend\n"
                "semantic_only,0.5,0.5,0.5,0.5,offline_heuristic\n"
                "bm25_only,1.0,1.0,1.0,1.0,gemini_free_tier_ragas\n"
                "hybrid_no_rerank,1.0,1.0,1.0,1.0,gemini_free_tier_ragas\n"
                "hybrid_rerank,1.0,1.0,1.0,1.0,gemini_free_tier_ragas\n",
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "USE_GEMINI_FREE_RAGAS": "1",
                        "ALLOW_CLOUD_FREE_TIER": "1",
                        "GEMINI_API_KEY": "test-key",
                    },
                    clear=True,
                ),
                patch.object(demo_eval, "evaluate", fake_evaluate),
                patch.object(demo_eval.evaluate, "RAGAS_RESULTS_PATH", results_path),
            ):
                status = demo_eval.run_cloud_eval_if_available()

        fake_evaluate.main.assert_called_once_with()
        self.assertEqual(status["status"], "skipped")
        self.assertIn("offline heuristics", status["reason"])

    def test_run_prepare_requires_huggingface_fetch_gate(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(demo_eval, "prepare_sources") as prepare,
        ):
            with self.assertRaisesRegex(RuntimeError, "ALLOW_HF_FETCH=1"):
                demo_eval.run_prepare()

        prepare.assert_not_called()

    def test_run_prepare_uses_environment_huggingface_fetch_gate(self):
        with (
            patch.dict(os.environ, {"ALLOW_HF_FETCH": "1"}, clear=True),
            patch.object(demo_eval, "prepare_sources", return_value=[Path("data/raw/source/README.md")]) as prepare,
        ):
            details = demo_eval.run_prepare()

        prepare.assert_called_once_with([demo_eval.HF_SOURCE])
        self.assertIn("prepared 1 file", details)

    def test_run_query_reuses_existing_index_and_disables_cloud_chat(self):
        fake_result = {"answer": "answer", "trace": {"synthesis": {"mode": "extractive"}}}
        with (
            patch.dict(os.environ, {"ALLOW_CLOUD_CHAT": "1"}, clear=True),
            patch.object(demo_eval, "chat_query", return_value=fake_result) as chat,
        ):
            details = demo_eval.run_query()
            self.assertEqual(os.environ["ALLOW_CLOUD_CHAT"], "1")

        chat.assert_called_once_with(
            demo_eval.DEMO_QUESTION,
            strategy="hybrid_rerank",
            allow_index_build=False,
        )
        self.assertIn("answer=answer", details)

    def test_run_demo_marks_cloud_eval_skipped_when_prereqs_missing(self):
        fake_results = [
            "prepared",
            "indexed",
            "answered",
            "offline eval",
        ]
        fake_cloud_status = {"status": "skipped", "reason": "Missing cloud prerequisites."}
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(demo_eval, "run_stage", side_effect=[
                demo_eval.StageResult("prepare", "completed", fake_results[0]),
                demo_eval.StageResult("index", "completed", fake_results[1]),
                demo_eval.StageResult("query", "completed", fake_results[2]),
                demo_eval.StageResult("offline_eval", "completed", fake_results[3]),
            ]),
            patch.object(demo_eval, "run_cloud_eval_if_available", return_value=fake_cloud_status),
        ):
            results = demo_eval.run_demo()

        self.assertEqual(results[-1].name, "cloud_eval")
        self.assertEqual(results[-1].status, "skipped")
        self.assertEqual(results[-1].details, fake_cloud_status["reason"])


if __name__ == "__main__":
    unittest.main()
