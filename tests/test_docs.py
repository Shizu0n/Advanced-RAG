import re
import unittest
from pathlib import Path


class DocumentationContractTests(unittest.TestCase):
    def test_env_example_documents_phase_6_runtime_flags(self):
        text = Path(".env.example").read_text(encoding="utf-8")

        required_names = [
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "GITHUB_MODELS_TOKEN",
            "GITHUB_MODELS_MODEL",
            "ALLOW_HF_FETCH",
            "ALLOW_INDEX_BUILD",
            "ALLOW_MODEL_DOWNLOADS",
            "ALLOW_GITHUB_FETCH",
            "ALLOW_DOCS_DOWNLOAD",
            "ALLOW_CLOUD_CHAT",
            "MAX_CLOUD_CHAT_CALLS",
            "CLOUD_CHAT_PROVIDER_TIMEOUT_SECONDS",
            "CLOUD_CHAT_TOTAL_TIMEOUT_SECONDS",
            "USE_GEMINI_FREE_RAGAS",
            "ALLOW_CLOUD_FREE_TIER",
            "MAX_GEMINI_CALLS",
            "GEMINI_RAGAS_STRICT",
        ]
        documented_keys = {
            match.group(1)
            for match in re.finditer(r"(?m)^(?!\s*#)\s*([A-Z0-9_]+)=", text)
        }

        for name in required_names:
            self.assertIn(name, documented_keys)

    def test_readme_uses_direct_professional_positioning(self):
        text = Path("README.md").read_text(encoding="utf-8")

        self.assertNotIn("Phase 6", text)
        self.assertNotIn("portfolio", text.lower())
        self.assertNotIn("Query log evidence", text)
        self.assertNotIn("scripts/demo_eval.py", text)
        self.assertNotIn("scripts/demo_evidence.py", text)

    def test_demo_artifacts_are_not_tracked(self):
        self.assertFalse(Path("scripts/demo_eval.py").exists())
        self.assertFalse(Path("scripts/demo_evidence.py").exists())
        self.assertFalse(Path("tests/test_demo_eval.py").exists())
        self.assertFalse(Path("tests/test_demo_evidence.py").exists())
        self.assertFalse(Path("assets/demo/phase6-query-trace.png").exists())
        self.assertFalse(Path("data/demo/phase6-query-trace.html").exists())
        self.assertFalse(Path("data/demo/phase6_demo_evidence.md").exists())

    def test_readme_documents_session_gate_overrides(self):
        text = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("session", text.lower())
        self.assertIn(".env", text)
        self.assertIn("ALLOW_CLOUD_CHAT", text)


if __name__ == "__main__":
    unittest.main()
