import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import requests
from llama_index.core.schema import TextNode

import evaluate as evaluation


class FailingProvider:
    name = "failing"

    def generate(self, node):
        raise RuntimeError("provider unavailable")


class StaticProvider:
    name = "static"

    def generate(self, node):
        return {
            "question": f"What does this specific chunk explain about {node.node_id}?",
            "ground_truth": node.get_content()[:80],
        }


class StaticPipeline:
    def answer_query(self, question, strategy):
        return {
            "answer": f"answer for {question}",
            "contexts": [f"context for {question}"],
        }


class GoldenDatasetTests(unittest.TestCase):
    def test_checked_in_golden_dataset_is_minimal_valid_and_covers_task_5_topics(self):
        dataset = evaluation.load_golden_dataset()

        self.assertGreaterEqual(len(dataset), 2)
        questions = " ".join(item["question"].lower() for item in dataset)
        sources = {item["source_doc"] for item in dataset}
        self.assertIn("abstract", questions)
        self.assertIn("stack", questions)
        self.assertIn("data/eval/project_abstract.md", sources)
        self.assertIn("data/eval/stack_discovery.md", sources)

    def test_run_evaluation_offline_never_calls_ragas_and_labels_backends_honestly(self):
        with TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden_dataset.json"
            summary_path = Path(tmpdir) / "summary.csv"
            detail_path = Path(tmpdir) / "detail.csv"
            golden_path.write_text(
                json.dumps(
                    [
                        {
                            "question": "What does the project abstract say?",
                            "ground_truth": "The project evaluates local RAG strategies offline.",
                            "reference_context": "The project evaluates local RAG strategies offline.",
                            "source_doc": "data/eval/project_abstract.md",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch.object(evaluation, "STRATEGIES", ["semantic_only"]),
                patch.object(evaluation, "EVAL_DIR", Path(tmpdir)),
                patch.object(evaluation, "RAGAS_RESULTS_PATH", summary_path),
                patch.object(evaluation, "RAGAS_PER_QUESTION_PATH", detail_path),
                patch.dict("os.environ", {}, clear=True),
                patch.object(evaluation.gemini_ragas, "run_ragas") as run_ragas,
            ):
                evaluation.run_evaluation(golden_path=golden_path, pipeline=StaticPipeline())

            run_ragas.assert_not_called()
            detail = pd.read_csv(detail_path)
            summary = pd.read_csv(summary_path)
            self.assertEqual(detail.loc[0, "evaluation_backend"], "offline_heuristic")
            self.assertEqual(detail.loc[0, "summary_backend"], "offline_heuristic")
            self.assertEqual(summary.loc[0, "summary_backend"], "offline_heuristic")

    def test_provider_fallback_uses_next_provider(self):
        node = TextNode(id_="n1", text="Python functions are defined with def and return values.")

        item = evaluation.generate_golden_item(node, [FailingProvider(), StaticProvider()])

        self.assertEqual(item["provider"], "static")
        self.assertIn("question", item)
        self.assertIn("ground_truth", item)
        self.assertEqual(item["source_doc"], "n1")

    def test_filter_keeps_most_specific_questions(self):
        candidates = [
            {
                "question": "What is Python?",
                "ground_truth": "short",
                "reference_context": "ctx",
                "source_doc": "a",
            },
            {
                "question": "How does Python manage default argument values in function definitions and calls?",
                "ground_truth": "specific",
                "reference_context": "ctx",
                "source_doc": "b",
            },
            {
                "question": "Explain decorators with arguments and wrapper function behavior in Python.",
                "ground_truth": "specific",
                "reference_context": "ctx",
                "source_doc": "c",
            },
        ]

        filtered = evaluation.filter_best_golden_items(candidates, limit=2)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["source_doc"], "b")
        self.assertEqual(filtered[1]["source_doc"], "c")

    def test_run_evaluation_rejects_empty_golden_dataset(self):
        with TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden_dataset.json"
            golden_path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-empty"):
                evaluation.run_evaluation(golden_path=golden_path, pipeline=object())

    def test_run_evaluation_rejects_dataset_missing_required_fields(self):
        with TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden_dataset.json"
            golden_path.write_text('[{"question": "What is missing?"}]', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "ground_truth"):
                evaluation.run_evaluation(golden_path=golden_path, pipeline=object())

    def test_main_regenerates_existing_invalid_golden_dataset_without_building_index_by_default(self):
        with TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden_dataset.json"
            golden_path.write_text("[]", encoding="utf-8")
            node = TextNode(id_="offline", text="Offline Python context explains functions.")

            with (
                patch.dict("os.environ", {}, clear=True),
                patch.object(evaluation, "GOLDEN_DATASET_PATH", golden_path),
                patch.object(evaluation, "build_index") as build,
                patch.object(evaluation, "load_local_context_nodes", return_value=[node]),
                patch.object(evaluation, "generate_golden_dataset") as generate,
                patch.object(evaluation, "LocalRAGPipeline", return_value="pipeline"),
                patch.object(evaluation, "run_evaluation") as run,
            ):
                evaluation.main()

            build.assert_not_called()
            generate.assert_called_once_with([node], output_path=golden_path)
            run.assert_called_once_with(pipeline="pipeline")

    def test_main_can_build_index_only_with_explicit_opt_in(self):
        with TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden_dataset.json"
            golden_path.write_text("[]", encoding="utf-8")

            with (
                patch.dict("os.environ", {"ALLOW_INDEX_BUILD": "1"}, clear=True),
                patch.object(evaluation, "GOLDEN_DATASET_PATH", golden_path),
                patch.object(evaluation, "build_index", return_value=("index", ["node"])) as build,
                patch.object(evaluation, "generate_golden_dataset") as generate,
                patch.object(evaluation, "LocalRAGPipeline", return_value="pipeline"),
                patch.object(evaluation, "run_evaluation") as run,
            ):
                evaluation.main()

            build.assert_called_once()
            generate.assert_called_once_with(["node"], output_path=golden_path)
            run.assert_called_once_with(pipeline="pipeline")

    def test_gemini_ragas_opt_in_requires_cloud_free_tier_and_key(self):
        rows = [
            {
                "question": "What is Python?",
                "answer": "Python is a language.",
                "contexts": '["Python is a language."]',
                "ground_truth": "Python is a language.",
            }
        ]

        with (
            patch.dict("os.environ", {"USE_GEMINI_FREE_RAGAS": "1"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "ALLOW_CLOUD_FREE_TIER"),
        ):
            evaluation.maybe_run_real_ragas(rows)

    def test_cloud_failures_fall_back_to_offline_unless_strict_mode_is_set(self):
        rows = [
            {
                "question": "What is Python?",
                "answer": "Python is a language.",
                "contexts": '["Python is a language."]',
                "ground_truth": "Python is a language.",
            }
        ]
        recoverable = [
            requests.exceptions.Timeout("timeout"),
            RuntimeError("MAX_GEMINI_CALLS exceeded"),
            RuntimeError("Gemini models unavailable"),
        ]

        for error in recoverable:
            with (
                self.subTest(error=type(error).__name__),
                patch.dict(
                    "os.environ",
                    {
                        "USE_GEMINI_FREE_RAGAS": "1",
                        "ALLOW_CLOUD_FREE_TIER": "1",
                        "GEMINI_API_KEY": "key",
                    },
                    clear=True,
                ),
                patch.object(evaluation.gemini_ragas, "run_ragas", side_effect=error),
            ):
                self.assertIsNone(evaluation.maybe_run_real_ragas(rows, gemini_client=object()))

        with (
            patch.dict(
                "os.environ",
                {
                    "USE_GEMINI_FREE_RAGAS": "1",
                    "ALLOW_CLOUD_FREE_TIER": "1",
                    "GEMINI_API_KEY": "key",
                    "GEMINI_RAGAS_STRICT": "1",
                },
                clear=True,
            ),
            patch.object(evaluation.gemini_ragas, "run_ragas", side_effect=ValueError("bad json")),
            self.assertRaisesRegex(ValueError, "bad json"),
        ):
            evaluation.maybe_run_real_ragas(rows, gemini_client=object())

    def test_integration_errors_do_not_fall_back_silently(self):
        rows = [
            {
                "question": "What is Python?",
                "answer": "Python is a language.",
                "contexts": '["Python is a language."]',
                "ground_truth": "Python is a language.",
            }
        ]

        with (
            patch.dict(
                "os.environ",
                {
                    "USE_GEMINI_FREE_RAGAS": "1",
                    "ALLOW_CLOUD_FREE_TIER": "1",
                    "GEMINI_API_KEY": "key",
                },
                clear=True,
            ),
            patch.object(evaluation.gemini_ragas, "run_ragas", side_effect=ValueError("schema bug")),
            self.assertRaisesRegex(ValueError, "schema bug"),
        ):
            evaluation.maybe_run_real_ragas(rows, gemini_client=object())

    def test_run_evaluation_reuses_one_gemini_client_across_strategies(self):
        with TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden_dataset.json"
            golden_path.write_text(
                json.dumps(
                    [
                        {
                            "question": "What is Python?",
                            "ground_truth": "Python is a language.",
                            "reference_context": "Python is a language.",
                            "source_doc": "doc.md",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            client = object()

            with (
                patch.object(evaluation, "STRATEGIES", ["semantic_only", "bm25_only"]),
                patch.object(evaluation, "EVAL_DIR", Path(tmpdir)),
                patch.object(evaluation, "RAGAS_RESULTS_PATH", Path(tmpdir) / "summary.csv"),
                patch.object(evaluation, "RAGAS_PER_QUESTION_PATH", Path(tmpdir) / "detail.csv"),
                patch.dict(
                    "os.environ",
                    {
                        "USE_GEMINI_FREE_RAGAS": "1",
                        "ALLOW_CLOUD_FREE_TIER": "1",
                        "GEMINI_API_KEY": "key",
                    },
                    clear=True,
                ),
                patch.object(evaluation.gemini_ragas, "client_from_config", return_value=client),
                patch.object(
                    evaluation.gemini_ragas,
                    "run_ragas",
                    return_value={
                        "faithfulness": 0.9,
                        "answer_relevancy": 0.8,
                        "context_recall": 0.7,
                        "context_precision": 0.6,
                    },
                ) as run_ragas,
            ):
                evaluation.run_evaluation(golden_path=golden_path, pipeline=StaticPipeline())

            clients = [
                call.kwargs["gemini_client"]
                for call in run_ragas.call_args_list
            ]
            self.assertEqual(clients, [client, client])

    def test_run_evaluation_keeps_per_question_backend_honest_when_summary_uses_ragas(self):
        with TemporaryDirectory() as tmpdir:
            golden_path = Path(tmpdir) / "golden_dataset.json"
            summary_path = Path(tmpdir) / "summary.csv"
            detail_path = Path(tmpdir) / "detail.csv"
            golden_path.write_text(
                json.dumps(
                    [
                        {
                            "question": "What is Python?",
                            "ground_truth": "Python is a language.",
                            "reference_context": "Python is a language.",
                            "source_doc": "doc.md",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            ragas_scores = {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_recall": 0.7,
                "context_precision": 0.6,
            }

            with (
                patch.object(evaluation, "STRATEGIES", ["semantic_only"]),
                patch.object(evaluation, "EVAL_DIR", Path(tmpdir)),
                patch.object(evaluation, "RAGAS_RESULTS_PATH", summary_path),
                patch.object(evaluation, "RAGAS_PER_QUESTION_PATH", detail_path),
                patch.dict(
                    "os.environ",
                    {
                        "USE_GEMINI_FREE_RAGAS": "1",
                        "ALLOW_CLOUD_FREE_TIER": "1",
                        "GEMINI_API_KEY": "key",
                    },
                    clear=True,
                ),
                patch.object(evaluation.gemini_ragas, "client_from_config", return_value=object()),
                patch.object(evaluation.gemini_ragas, "run_ragas", return_value=ragas_scores),
            ):
                evaluation.run_evaluation(golden_path=golden_path, pipeline=StaticPipeline())

            detail = pd.read_csv(detail_path)
            summary = pd.read_csv(summary_path)
            self.assertEqual(detail.loc[0, "evaluation_backend"], "offline_heuristic")
            self.assertEqual(detail.loc[0, "summary_backend"], "gemini_free_tier_ragas")
            self.assertEqual(summary.loc[0, "summary_backend"], "gemini_free_tier_ragas")

    def test_requirements_exclude_direct_paid_provider_dependencies(self):
        requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
        forbidden = ["openai", "anthropic", "langchain-openai", "langchain-anthropic"]

        for package in forbidden:
            self.assertNotIn(package, requirements)

    def test_requirements_pin_matches_locally_tested_ragas_version(self):
        requirements = Path("requirements.txt").read_text(encoding="utf-8")
        self.assertIn("ragas==0.2.5", requirements)


if __name__ == "__main__":
    unittest.main()
