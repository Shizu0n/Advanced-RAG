import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import cloud_ragas


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class CloudRagasTests(unittest.TestCase):
    def test_config_requires_explicit_free_tier_opt_in_and_at_least_one_provider(self):
        with patch.dict("os.environ", {"USE_CLOUD_FREE_TIER_RAGAS": "1", "GEMINI_API_KEY": "key"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ALLOW_CLOUD_FREE_TIER"):
                cloud_ragas.config_from_env()

        with patch.dict(
            "os.environ",
            {"USE_CLOUD_FREE_TIER_RAGAS": "1", "ALLOW_CLOUD_FREE_TIER": "1"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "Set at least one supported free-tier cloud provider"):
                cloud_ragas.config_from_env()

        with patch.dict(
            "os.environ",
            {"USE_CLOUD_FREE_TIER_RAGAS": "1", "ALLOW_CLOUD_FREE_TIER": "1", "GROQ_API_KEY": "groq"},
            clear=True,
        ):
            config = cloud_ragas.config_from_env()

        self.assertEqual([provider.name for provider in config.providers], ["groq"])

    def test_ragas_evaluate_receives_explicit_llm_and_embeddings(self):
        rows = [
            {
                "question": "What is Python?",
                "answer": "Python is a language.",
                "contexts": '["Python is a language."]',
                "ground_truth": "Python is a language.",
            }
        ]
        fake_result = Mock()
        fake_result.to_pandas.return_value = Mock(to_dict=Mock(return_value={"faithfulness": {0: 1.0}}))

        with (
            patch.object(cloud_ragas, "_import_ragas_parts") as imports,
            patch.object(cloud_ragas, "build_cloud_backends", return_value=("llm", "embeddings")),
        ):
            imports.return_value = {
                "Dataset": Mock(from_list=Mock(return_value="dataset")),
                "evaluate": Mock(return_value=fake_result),
                "metrics": ["metric"],
            }

            scores = cloud_ragas.run_ragas(rows)

        imports.return_value["Dataset"].from_list.assert_called_once_with(
            [
                {
                    "user_input": "What is Python?",
                    "response": "Python is a language.",
                    "retrieved_contexts": ["Python is a language."],
                    "reference": "Python is a language.",
                }
            ]
        )
        imports.return_value["evaluate"].assert_called_once_with(
            "dataset",
            metrics=["metric"],
            llm="llm",
            embeddings="embeddings",
            batch_size=1,
            raise_exceptions=True,
            show_progress=False,
        )
        self.assertEqual(scores["faithfulness"], 1.0)

    def test_ragas_record_uses_ragas_0_2_required_columns(self):
        row = {
            "question": "What is Python?",
            "answer": "Python is a language.",
            "contexts": ["Python is a language."],
            "ground_truth": "Python is a language.",
        }

        self.assertEqual(
            cloud_ragas._ragas_record(row),
            {
                "user_input": "What is Python?",
                "response": "Python is a language.",
                "retrieved_contexts": ["Python is a language."],
                "reference": "Python is a language.",
            },
        )

    def test_provider_quota_error_sets_short_cooldown_for_next_request(self):
        calls = []

        def post(url, json, timeout, **kwargs):
            calls.append(url)
            if "generativelanguage.googleapis.com" in url:
                return FakeResponse(429, {"error": {"message": "quota"}})
            return FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})

        with TemporaryDirectory() as tmpdir:
            client = cloud_ragas.FreeTierCloudClient(
                cache_dir=Path(tmpdir),
                post=post,
                max_calls=5,
                providers=[
                    cloud_ragas.CloudProvider("gemini", "gemini-2.5-flash", "key"),
                    cloud_ragas.CloudProvider("groq", "llama-3.3-70b-versatile", "groq-key"),
                ],
            )
            client.generate_text("first uncached prompt")
            client.generate_text("second uncached prompt")

        self.assertEqual(sum("generativelanguage.googleapis.com" in call for call in calls), 1)
        self.assertEqual(sum("api.groq.com" in call for call in calls), 2)

    def test_ragas_llm_reports_unfinished_empty_generation(self):
        with patch.object(cloud_ragas, "FreeTierCloudClient"):
            llm, _ = cloud_ragas.build_cloud_backends(cloud_client=object())

        response = SimpleNamespace(generations=[[SimpleNamespace(text="")]])

        self.assertFalse(llm.is_finished(response))

    def test_model_fallback_order_skips_quota_errors(self):
        calls = []
        headers_seen = []

        def post(url, json, timeout, **kwargs):
            calls.append(url)
            headers_seen.append(kwargs.get("headers", {}))
            if "generativelanguage.googleapis.com" in url:
                return FakeResponse(429, {"error": {"message": "quota"}})
            return FakeResponse(
                200,
                {"choices": [{"message": {"content": '{"question":"Q?","ground_truth":"A."}'}}]},
            )

        with TemporaryDirectory() as tmpdir:
            client = cloud_ragas.FreeTierCloudClient(
                cache_dir=Path(tmpdir),
                post=post,
                max_calls=5,
                providers=[
                    cloud_ragas.CloudProvider("gemini", "gemini-2.5-flash", "key"),
                    cloud_ragas.CloudProvider("groq", "llama-3.3-70b-versatile", "groq-key"),
                ],
            )

            data = client.generate_json("prompt")

        self.assertEqual(data, {"question": "Q?", "ground_truth": "A."})
        self.assertIn("gemini-2.5-flash", calls[0])
        self.assertNotIn("key=", calls[0])
        self.assertEqual(headers_seen[0]["x-goog-api-key"], "key")
        self.assertIn("api.groq.com", calls[1])

    def test_provider_chain_uses_configured_allowlisted_order(self):
        with patch.dict(
            "os.environ",
            {
                "CLOUD_PROVIDER_ORDER": "groq,gemini,github",
                "GEMINI_API_KEY": "gemini",
                "GEMINI_MODEL": "gemini-2.5-pro",
                "GROQ_API_KEY": "groq",
                "GROQ_MODEL": "llama-3.3-70b-versatile",
                "GITHUB_MODELS_TOKEN": "github",
                "GITHUB_MODELS_MODEL": "openai/gpt-4o-mini",
            },
            clear=True,
        ):
            providers = cloud_ragas.providers_from_env()

        self.assertEqual([provider.name for provider in providers], ["groq", "gemini", "github"])
        self.assertEqual(providers[1].model, "gemini-2.5-pro")

    def test_provider_order_rejects_unsupported_provider_names(self):
        with patch.dict("os.environ", {"CLOUD_PROVIDER_ORDER": "gemini,custom", "GEMINI_API_KEY": "key"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Unsupported CLOUD_PROVIDER_ORDER provider: custom"):
                cloud_ragas.providers_from_env()

    def test_provider_chain_skips_google_family_fallback_models(self):
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "gemini",
                "GROQ_API_KEY": "groq",
                "GROQ_MODEL": "gemma2-9b-it",
                "GITHUB_MODELS_TOKEN": "github",
                "GITHUB_MODELS_MODEL": "google/gemini-2.5-pro",
            },
            clear=True,
        ):
            providers = cloud_ragas.providers_from_env()

        self.assertEqual([provider.name for provider in providers], ["gemini"])

    def test_cache_prevents_repeat_calls_and_max_guard_blocks_uncached_calls(self):
        post = Mock(
            return_value=FakeResponse(
                200,
                {"embedding": {"values": [0.1, 0.2, 0.3]}},
            )
        )

        with TemporaryDirectory() as tmpdir:
            client = cloud_ragas.FreeTierCloudClient(
                cache_dir=Path(tmpdir),
                post=post,
                max_calls=1,
                providers=[cloud_ragas.CloudProvider("gemini", "gemini-2.5-flash", "key")],
            )

            self.assertEqual(client.embed_text("same text"), [0.1, 0.2, 0.3])
            self.assertEqual(client.embed_text("same text"), [0.1, 0.2, 0.3])
            with self.assertRaisesRegex(RuntimeError, "MAX_CLOUD_CALLS"):
                client.embed_text("new text")

        self.assertEqual(post.call_count, 1)

    def test_shared_budget_applies_across_clients_without_counting_cache_hits(self):
        post = Mock(
            return_value=FakeResponse(
                200,
                {"embedding": {"values": [0.1, 0.2, 0.3]}},
            )
        )

        with TemporaryDirectory() as tmpdir:
            budget = cloud_ragas.CloudCallBudget(max_calls=1)
            first = cloud_ragas.FreeTierCloudClient(
                cache_dir=Path(tmpdir),
                post=post,
                budget=budget,
                providers=[cloud_ragas.CloudProvider("gemini", "gemini-2.5-flash", "key")],
            )
            second = cloud_ragas.FreeTierCloudClient(
                cache_dir=Path(tmpdir),
                post=post,
                budget=budget,
                providers=[cloud_ragas.CloudProvider("gemini", "gemini-2.5-flash", "key")],
            )

            self.assertEqual(first.embed_text("same text"), [0.1, 0.2, 0.3])
            self.assertEqual(second.embed_text("same text"), [0.1, 0.2, 0.3])
            with self.assertRaisesRegex(RuntimeError, "MAX_CLOUD_CALLS"):
                second.embed_text("new text")

        self.assertEqual(post.call_count, 1)

    def test_source_has_no_paid_provider_calls_or_config(self):
        source = Path("cloud_ragas.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("openai_api_key", source)
        self.assertNotIn("anthropic", source)


if __name__ == "__main__":
    unittest.main()
