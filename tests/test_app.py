import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import app


class FakeContext:
    def __init__(self, enter=None):
        self.enter = enter

    def __enter__(self):
        if self.enter:
            self.enter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeStreamlit:
    def __init__(self, prompt=None, messages=None):
        self.session_state = {}
        if messages is not None:
            self.session_state[app.CHAT_MESSAGES_KEY] = messages
        self.prompt = prompt
        self.events = []

    def selectbox(self, label, options, index=0):
        self.events.append(("selectbox", label, options[index]))
        return options[index]

    def chat_input(self, label):
        self.events.append(("chat_input", label))
        return self.prompt

    def chat_message(self, role):
        return FakeContext(lambda: self.events.append(("chat_message", role)))

    def write(self, value):
        self.events.append(("write", value))

    def caption(self, value):
        self.events.append(("caption", value))

    def expander(self, label, expanded=False):
        return FakeContext(lambda: self.events.append(("expander", label, expanded)))

    def dataframe(self, value, use_container_width=False):
        self.events.append(("dataframe", list(value.columns), use_container_width))

    def json(self, value):
        self.events.append(("json", value))

    def error(self, value):
        self.events.append(("error", value))


class AppHelperTests(unittest.TestCase):
    def test_load_eval_summary_reads_expected_metric_columns(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ragas_results.csv"
            path.write_text(
                "strategy,faithfulness,answer_relevancy,context_recall,context_precision,summary_backend\n"
                "semantic_only,0.1,0.2,0.3,0.4,gemini_free_tier_ragas\n",
                encoding="utf-8",
            )

            frame = app.load_eval_summary(path)

        self.assertEqual(frame.loc[0, "strategy"], "semantic_only")
        self.assertEqual(frame.loc[0, "faithfulness"], 0.1)
        self.assertEqual(frame.loc[0, "summary_backend"], "gemini_free_tier_ragas")
        self.assertEqual(list(frame.columns), ["strategy", *app.METRICS, "summary_backend"])

    def test_load_per_question_handles_empty_csv_file(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ragas_per_question.csv"
            path.write_text("", encoding="utf-8")

            frame = app.load_per_question(path)

        self.assertTrue(frame.empty)
        self.assertIn("strategy", frame.columns)
        self.assertIn("question", frame.columns)

    def test_best_strategy_uses_mean_metric_score(self):
        frame = pd.DataFrame(
            [
                {
                    "strategy": "bm25_only",
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.1,
                    "context_recall": 0.1,
                    "context_precision": 0.1,
                },
                {
                    "strategy": "hybrid_rerank",
                    "faithfulness": 0.5,
                    "answer_relevancy": 0.5,
                    "context_recall": 0.5,
                    "context_precision": 0.5,
                },
            ]
        )

        self.assertEqual(app.best_strategy(frame), "hybrid_rerank")

    def test_metric_card_values_default_to_best_strategy(self):
        frame = pd.DataFrame(
            [
                {
                    "strategy": "weak",
                    "faithfulness": 0.1,
                    "answer_relevancy": 0.1,
                    "context_recall": 0.1,
                    "context_precision": 0.1,
                },
                {
                    "strategy": "strong",
                    "faithfulness": 0.7,
                    "answer_relevancy": 0.8,
                    "context_recall": 0.9,
                    "context_precision": 1.0,
                },
            ]
        )

        values = app.metric_card_values(frame)

        self.assertEqual(values["strategy"], "strong")
        self.assertEqual(values["context_precision"], 1.0)

    def test_filter_questions_below_threshold_keeps_any_low_metric(self):
        frame = pd.DataFrame(
            [
                {"question": "keep", "faithfulness": 0.9, "answer_relevancy": 0.2},
                {"question": "drop", "faithfulness": 0.9, "answer_relevancy": 0.9},
            ]
        )

        filtered = app.filter_questions_below_threshold(frame, 0.5)

        self.assertEqual(filtered["question"].tolist(), ["keep"])

    def test_normalize_trace_extracts_score_lists_from_common_keys(self):
        trace = {
            "bm25_scores": [{"source": "a", "score": 0.7}],
            "vector_results": [{"source_doc": "b", "score": 0.6}],
            "rrf_scores": [("c", 0.5)],
            "reranker_scores": {"d": 0.4},
            "used_rerank": True,
        }

        normalized = app.normalize_trace(trace)

        self.assertEqual(normalized["bm25_scores"], [{"source": "a", "score": 0.7}])
        self.assertEqual(normalized["vector_scores"], [{"source": "b", "score": 0.6}])
        self.assertEqual(normalized["rrf_scores"], [{"source": "c", "score": 0.5}])
        self.assertEqual(normalized["reranker_scores"], [{"source": "d", "score": 0.4}])
        self.assertEqual(normalized["metadata"]["used_rerank"], True)

    def test_run_query_forces_offline_safe_pipeline_mode(self):
        with patch("pipeline.answer_query", return_value={"answer": "ok"}) as query:
            result = app.run_query("question", "bm25_only")

        self.assertEqual(result["answer"], "ok")
        query.assert_called_once_with("question", strategy="bm25_only", allow_index_build=False)

    def test_run_chat_query_forces_offline_safe_pipeline_mode(self):
        history = [{"role": "user", "content": "before"}]
        with patch("pipeline.chat_query", return_value={"answer": "ok"}) as query:
            result = app.run_chat_query("question", history, "bm25_only")

        self.assertEqual(result["answer"], "ok")
        query.assert_called_once_with(
            "question",
            history=history,
            strategy="bm25_only",
            allow_index_build=False,
        )

    def test_query_tab_persists_chat_messages_in_session_state(self):
        fake_st = FakeStreamlit(prompt="Qual e a stack?")
        response = {
            "answer": "React e Vite.",
            "citations": [{"source_doc": "README.md", "score": 0.9, "snippet": "React 19 e Vite 7."}],
            "trace": {"strategy": "hybrid_rerank"},
        }

        with patch("app.run_chat_query", return_value=response) as query:
            app._render_query_tab(fake_st)

        messages = fake_st.session_state[app.CHAT_MESSAGES_KEY]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["content"], "Qual e a stack?")
        self.assertEqual(messages[1]["content"], "React e Vite.")
        query.assert_called_once_with("Qual e a stack?", history=[], strategy="hybrid_rerank")

    def test_query_tab_sends_existing_session_history_to_chat_query(self):
        existing = [{"role": "user", "content": "contexto anterior"}]
        fake_st = FakeStreamlit(prompt="continua", messages=existing)

        with patch("app.run_chat_query", return_value={"answer": "ok", "citations": [], "trace": {}}) as query:
            app._render_query_tab(fake_st)

        query.assert_called_once_with("continua", history=[{"role": "user", "content": "contexto anterior"}], strategy="hybrid_rerank")
        self.assertEqual(len(fake_st.session_state[app.CHAT_MESSAGES_KEY]), 3)

    def test_assistant_message_renders_citations_and_keeps_debug_separate(self):
        fake_st = FakeStreamlit()
        message = {
            "role": "assistant",
            "content": "Resposta curta.",
            "citations": [{"source_doc": "README.md", "score": 0.81234, "snippet": "Trecho usado como evidencia."}],
            "trace": {"bm25_scores": [{"source": "README.md", "score": 0.8}], "retrieval_query": "q"},
        }

        app._render_chat_message(fake_st, message)

        captions = [event[1] for event in fake_st.events if event[0] == "caption"]
        expanders = [event[1] for event in fake_st.events if event[0] == "expander"]
        json_payloads = [event[1] for event in fake_st.events if event[0] == "json"]
        self.assertTrue(any("README.md score=0.812" in caption for caption in captions))
        self.assertIn("Retrieval trace / debug", expanders)
        self.assertEqual(json_payloads, [{"retrieval_query": "q"}])

    def test_prepare_sources_for_app_uses_explicit_github_opt_in(self):
        with patch("source_loader.prepare_sources", return_value=[Path("data/raw/repo/a.py")]) as prepare:
            files = app.prepare_sources_for_app(["https://github.com/user/repo"], allow_github_fetch=True)

        self.assertEqual(files, [Path("data/raw/repo/a.py")])
        prepare.assert_called_once_with(["https://github.com/user/repo"], allow_github_fetch=True)

    def test_eval_backend_counts_summarize_summary_and_question_backends(self):
        summary = pd.DataFrame(
            [
                {"strategy": "semantic_only", "summary_backend": "offline_heuristic"},
                {"strategy": "hybrid_rerank", "summary_backend": "gemini_free_tier_ragas"},
            ]
        )
        per_question = pd.DataFrame(
            [
                {"evaluation_backend": "offline_heuristic"},
                {"evaluation_backend": "offline_heuristic"},
            ]
        )

        counts = app.eval_backend_counts(summary, per_question)

        self.assertEqual(counts["summary_backends"]["gemini_free_tier_ragas"], 1)
        self.assertEqual(counts["question_backends"]["offline_heuristic"], 2)

    def test_default_eval_paths_are_project_local_when_cwd_changes(self):
        old_cwd = os.getcwd()
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "data", "eval").mkdir(parents=True)
            Path(tmpdir, "data", "eval", "ragas_results.csv").write_text(
                "strategy,faithfulness,answer_relevancy,context_recall,context_precision\n"
                "wrong,1,1,1,1\n",
                encoding="utf-8",
            )
            try:
                os.chdir(tmpdir)
                frame = app.load_eval_summary()
            finally:
                os.chdir(old_cwd)

        self.assertNotIn("wrong", frame["strategy"].tolist())


if __name__ == "__main__":
    unittest.main()
