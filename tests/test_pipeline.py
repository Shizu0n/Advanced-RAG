import unittest
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llama_index.core.schema import NodeWithScore, TextNode

import pipeline as pipeline_module
from pipeline import LocalRAGPipeline, answer_query


class FakeRetriever:
    def ablation_retrieve(self, query, strategy):
        nodes = [
            NodeWithScore(
                node=TextNode(
                    id_="doc-1",
                    text="Python functions are defined with the def keyword. They can return values.",
                    metadata={"file_name": "functions.txt"},
                ),
                score=0.9,
            )
        ]
        return nodes, {"strategy": strategy, "used_rerank": False}


class EmptyRetriever:
    def ablation_retrieve(self, query, strategy):
        return [], {"strategy": strategy}


class IrrelevantRetriever:
    def ablation_retrieve(self, query, strategy):
        nodes = [
            NodeWithScore(
                node=TextNode(
                    id_="doc-irrelevant",
                    text="Bananas maduras ficam amarelas em temperatura ambiente.",
                    metadata={"file_name": "bananas.txt"},
                ),
                score=0.01,
            )
        ]
        return nodes, {"strategy": strategy}


class PipelineTests(unittest.TestCase):
    def test_local_context_nodes_include_nested_repo_files(self):
        with TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "my-repo"
            source_dir = repo / "src"
            source_dir.mkdir(parents=True)
            (source_dir / "app.py").write_text(
                "def greet():\n    return 'hello from repo'\n",
                encoding="utf-8",
            )
            (repo / "package-lock.json").write_text(
                '{"name":"backend","packages":{"":{"dependencies":{"@nestjs/common":"^11.0.1"}}}}\n',
                encoding="utf-8",
            )
            (repo / "node_modules").mkdir()
            (repo / "node_modules" / "ignored.js").write_text("console.log('ignore me')", encoding="utf-8")

            nodes = pipeline_module.load_local_context_nodes([repo])

        texts = [node.text for node in nodes]
        sources = [node.metadata["file_name"] for node in nodes]

        self.assertTrue(any("hello from repo" in text for text in texts))
        self.assertTrue(any("src/app.py" in source for source in sources))
        self.assertFalse(any("node_modules" in source for source in sources))
        self.assertFalse(any("package-lock.json" in source for source in sources))

    def test_package_json_is_summarized_for_project_questions(self):
        with TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "backend"
            repo.mkdir()
            (repo / "package.json").write_text(
                """
                {
                  "name": "backend",
                  "version": "0.0.1",
                  "dependencies": {
                    "@nestjs/common": "^11.0.1",
                    "@nestjs/core": "^11.0.1",
                    "@nestjs/platform-express": "^11.0.1",
                    "@nestjs/typeorm": "^11.0.0",
                    "sqlite3": "^5.1.7"
                  },
                  "devDependencies": {
                    "typescript": "^5.7.3"
                  }
                }
                """,
                encoding="utf-8",
            )

            pipeline = LocalRAGPipeline(nodes=pipeline_module.load_local_context_nodes([repo]))
            result = pipeline.answer_query("What backend framework does this project use?", strategy="bm25_only")

        self.assertIn("NestJS", result["answer"])
        self.assertIn("Express", result["answer"])
        self.assertLess(len(result["answer"]), 800)
        self.assertNotIn('"packages"', result["answer"])

    def test_package_json_detects_frontend_frameworks_and_tooling(self):
        with TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "frontend"
            repo.mkdir()
            (repo / "package.json").write_text(
                """
                {
                  "name": "frontend",
                  "version": "1.0.0",
                  "dependencies": {
                    "@vitejs/plugin-react": "^5.0.0",
                    "axios": "^1.7.9",
                    "react": "^19.0.0",
                    "react-router-dom": "^7.1.1",
                    "vite": "^7.0.0"
                  },
                  "devDependencies": {
                    "@eslint/js": "^9.18.0",
                    "eslint": "^9.18.0",
                    "jest": "^29.7.0",
                    "prettier": "^3.4.2",
                    "typescript": "^5.7.3"
                  }
                }
                """,
                encoding="utf-8",
            )

            nodes = pipeline_module.load_local_context_nodes([repo])

        text = nodes[0].text
        for hint in ["Vite", "React", "React Router", "Axios", "TypeScript", "ESLint", "Jest", "Prettier"]:
            self.assertIn(hint, text)
        self.assertNotIn('{"name"', text)

    def test_readme_is_loaded_as_structured_section_snippets(self):
        with TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "README.md").write_text(
                """
# Project Alpha

Intro paragraph.

## Stack

React and Vite.

### Frontend

Axios and React Router.

## Setup

npm install.
                """.strip(),
                encoding="utf-8",
            )

            nodes = pipeline_module.load_local_context_nodes([repo / "README.md"])

        texts = [node.text for node in nodes]
        self.assertTrue(any("Section: Project Alpha" in text and "Intro paragraph" in text for text in texts))
        self.assertTrue(any("Section: Project Alpha > Stack" in text and "React and Vite" in text for text in texts))
        self.assertTrue(any("Section: Project Alpha > Stack > Frontend" in text and "Axios" in text for text in texts))
        self.assertTrue(any("Section: Project Alpha > Setup" in text and "npm install" in text for text in texts))

    def test_generic_json_context_is_summarized_not_dumped_as_answer(self):
        raw_json = '{"name":"alpha","secrets":{"token":"abc123"},"items":[{"id":1},{"id":2}]}'
        answer = pipeline_module.synthesize_extractive_answer("What JSON fields exist?", [raw_json])

        self.assertIn("JSON document summary", answer)
        self.assertIn("Top-level keys", answer)
        self.assertNotIn('{"name"', answer)
        self.assertNotIn('"token":"abc123"', answer)

    def test_generic_json_file_is_loaded_as_summary(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text('{"name":"alpha","enabled":true,"nested":{"debug":false}}', encoding="utf-8")

            nodes = pipeline_module.load_local_context_nodes([path])

        self.assertEqual(len(nodes), 1)
        self.assertIn("JSON document summary for config.json", nodes[0].text)
        self.assertIn("Top-level keys: enabled, name, nested", nodes[0].text)
        self.assertNotIn('{"name"', nodes[0].text)

    def test_answer_query_returns_extractive_shape(self):
        pipeline = LocalRAGPipeline(index=None, nodes=[], retriever=FakeRetriever())

        result = pipeline.answer_query("How are Python functions defined?", strategy="bm25_only")

        self.assertIn("def keyword", result["answer"])
        self.assertEqual(result["strategy"], "bm25_only")
        self.assertEqual(result["sources"][0]["source_doc"], "functions.txt")
        self.assertEqual(result["contexts"][0], result["sources"][0]["text"])
        self.assertEqual(result["trace"]["strategy"], "bm25_only")
        for key in ["bm25_scores", "vector_scores", "rrf_scores", "reranker_scores"]:
            self.assertIn(key, result["trace"])

    def test_default_answer_query_uses_local_lexical_fallback_without_index_build_opt_in(self):
        with patch.dict("os.environ", {}, clear=True):
            result = answer_query("What is the free-tier RAG workspace?", strategy="hybrid_rerank")

        self.assertIn("Free-tier RAG workspace", result["answer"])
        self.assertEqual(result["sources"][0]["source_doc"], "README.md")
        self.assertEqual(result["trace"]["fallback"], "local_lexical")
        self.assertTrue(result["trace"]["lexical_scores"])
        self.assertEqual(result["trace"]["bm25_scores"], [])
        self.assertEqual(result["trace"]["rrf_scores"], [])
        self.assertEqual(result["trace"]["vector_scores"], [])
        self.assertEqual(result["trace"]["reranker_scores"], [])

    def test_default_answer_query_returns_clear_empty_state_when_no_local_docs_exist(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(pipeline_module, "LOCAL_CONTEXT_PATHS", []),
        ):
            result = answer_query("What exists locally?", strategy="bm25_only")

        self.assertEqual(result["answer"], "No local context files were found for offline retrieval.")
        self.assertEqual(result["sources"], [])
        for key in ["bm25_scores", "vector_scores", "rrf_scores", "reranker_scores"]:
            self.assertIn(key, result["trace"])

    def test_answer_query_can_ignore_ambient_index_build_environment(self):
        fake_ingestion = SimpleNamespace(build_index=lambda: (_ for _ in ()).throw(AssertionError("must not build index")))
        with patch.dict("os.environ", {"ALLOW_INDEX_BUILD": "1"}, clear=True):
            with patch.object(pipeline_module, "load_local_context_nodes", return_value=[]):
                with patch.dict(sys.modules, {"ingestion": fake_ingestion}):
                    result = answer_query("What exists locally?", strategy="bm25_only", allow_index_build=False)

        self.assertEqual(result["answer"], "No local context files were found for offline retrieval.")

    def test_local_context_paths_are_project_local_when_cwd_changes(self):
        old_cwd = os.getcwd()
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "README.md").write_text("Unrelated cwd-only document.", encoding="utf-8")
            try:
                os.chdir(tmpdir)
                result = answer_query("What is the free-tier RAG workspace?", strategy="bm25_only")
            finally:
                os.chdir(old_cwd)

        self.assertIn("Free-tier RAG workspace", result["answer"])
        self.assertEqual(result["sources"][0]["source_doc"], "README.md")

    def test_trace_scores_are_derived_from_results_when_retriever_metadata_lacks_lists(self):
        pipeline = LocalRAGPipeline(index=None, nodes=[], retriever=FakeRetriever())

        result = pipeline.answer_query("How are Python functions defined?", strategy="bm25_only")

        self.assertEqual(result["trace"]["bm25_scores"], [{"source": "functions.txt", "score": 0.9}])
        self.assertEqual(result["trace"]["rrf_scores"], [])
        self.assertEqual(result["trace"]["vector_scores"], [])
        self.assertEqual(result["trace"]["reranker_scores"], [])

    def test_local_fallback_traces_intent_rewrite_and_does_not_fake_rrf_or_rerank(self):
        nodes = [
            pipeline_module.LocalTextNode(
                text="## Stack e Ferramentas\n\n### Frontend\n\n- React 19\n- Vite 7",
                node_id="readme#0",
                metadata={"file_name": "README.md"},
            )
        ]
        pipeline = LocalRAGPipeline(nodes=nodes)

        result = pipeline.answer_query("Qual stack do front?", strategy="hybrid_rerank")

        self.assertEqual(result["trace"]["intents"], ["stack"])
        self.assertIn("frontend", result["trace"]["rewritten_query"])
        self.assertTrue(result["trace"]["lexical_scores"])
        self.assertEqual(result["trace"]["rrf_scores"], [])
        self.assertEqual(result["trace"]["reranker_scores"], [])
        self.assertFalse(result["trace"]["used_rerank"])

    def test_stopwords_do_not_make_generic_documents_win(self):
        generic = pipeline_module.LocalTextNode(
            text="the the the is is de do da para com uma um projeto sistema",
            node_id="generic#0",
            metadata={"file_name": "generic.md"},
        )
        specific = pipeline_module.LocalTextNode(
            text="Frontend stack uses React with Vite and TypeScript.",
            node_id="specific#0",
            metadata={"file_name": "frontend/package.json"},
        )
        pipeline = LocalRAGPipeline(nodes=[generic, specific])

        result = pipeline.answer_query("Qual é a stack do frontend?", strategy="bm25_only")

        self.assertEqual(result["sources"][0]["source_doc"], "frontend/package.json")

    def test_frontend_stack_prioritizes_readme_stack_and_frontend_package_over_api_env_setup(self):
        readme = pipeline_module.LocalTextNode(
            text="## Stack e Ferramentas\n\n### Frontend\n\n- React 19\n- TypeScript 5\n- Vite 7",
            node_id="readme#0",
            metadata={"file_name": "README.md"},
        )
        package = pipeline_module.LocalTextNode(
            text="Package manifest for frontend package frontend with framework and platform hints: React. Runtime dependencies: react, vite.",
            node_id="package#0",
            metadata={"file_name": "frontend/package.json"},
        )
        api = pipeline_module.LocalTextNode(
            text="Frontend api client uses VITE_API_URL and setup env constants for requests.",
            node_id="api#0",
            metadata={"file_name": "frontend/src/services/api.ts"},
        )
        env = pipeline_module.LocalTextNode(
            text="VITE_API_URL=http://localhost:3000 frontend setup env",
            node_id="env#0",
            metadata={"file_name": "frontend/.env.example"},
        )
        pipeline = LocalRAGPipeline(nodes=[api, env, package, readme])

        result = pipeline.answer_query("What frontend tech stack does this use?", strategy="hybrid_no_rerank")
        top_sources = [source["source_doc"] for source in result["sources"][:2]]

        self.assertEqual(top_sources, ["README.md", "frontend/package.json"])

    def test_chat_query_returns_contract_with_citations(self):
        pipeline = LocalRAGPipeline(index=None, nodes=[], retriever=FakeRetriever())

        result = pipeline.chat_query("How are Python functions defined?", strategy="bm25_only")

        self.assertEqual(
            sorted(result.keys()),
            ["answer", "citations", "confidence", "contexts", "intent", "sources", "trace"],
        )
        self.assertIn("def keyword", result["answer"])
        self.assertEqual(result["intent"], "general")
        self.assertGreater(result["confidence"], 0)
        self.assertEqual(result["citations"][0]["source_doc"], "functions.txt")
        self.assertEqual(result["citations"][0]["score"], 0.9)
        self.assertIn("def keyword", result["citations"][0]["snippet"])
        self.assertNotIn("text", result["citations"][0])

    def test_chat_query_stack_aggregates_technologies_with_evidence(self):
        nodes = [
            pipeline_module.LocalTextNode(
                text="## Stack e Ferramentas\n\n### Frontend\n\n- React 19\n- TypeScript 5\n- Vite 7",
                node_id="readme#0",
                metadata={"file_name": "README.md"},
            ),
            pipeline_module.LocalTextNode(
                text="Package manifest for backend package backend with framework and platform hints: NestJS, Express, TypeORM, SQLite.",
                node_id="backend#0",
                metadata={"file_name": "backend/package.json"},
            ),
        ]
        pipeline = LocalRAGPipeline(nodes=nodes)

        result = pipeline.chat_query("Qual é a stack do projeto?", strategy="hybrid_rerank")

        self.assertEqual(result["intent"], "stack")
        for tech in ["React", "TypeScript", "Vite", "NestJS", "Express", "TypeORM", "SQLite"]:
            self.assertIn(tech, result["answer"])
        self.assertIn("Evidencia em:", result["answer"])
        self.assertTrue(all({"source_doc", "score", "snippet"} <= set(citation) for citation in result["citations"]))

    def test_chat_query_frontend_stack_does_not_mix_backend_technologies(self):
        nodes = [
            pipeline_module.LocalTextNode(
                text="### Frontend\n\n- React 19\n- TypeScript 5\n- Vite 7\n- React Router DOM 7\n- Axios",
                node_id="frontend#0",
                metadata={"file_name": "frontend/package.json"},
            ),
            pipeline_module.LocalTextNode(
                text="### Backend\n\n- NestJS 11\n- TypeORM 0.3\n- SQLite3",
                node_id="backend#0",
                metadata={"file_name": "backend/package.json"},
            ),
        ]
        pipeline = LocalRAGPipeline(nodes=nodes)

        result = pipeline.chat_query("quais sao as stacks do frontend", strategy="hybrid_rerank")

        for tech in ["React", "TypeScript", "Vite", "React Router", "Axios"]:
            self.assertIn(tech, result["answer"])
        for tech in ["NestJS", "TypeORM", "SQLite"]:
            self.assertNotIn(tech, result["answer"])

    def test_chat_query_overview_prioritizes_summary_with_evidence(self):
        nodes = [
            pipeline_module.LocalTextNode(
                text=(
                    "README section from README.md. Section: Advanced RAG Workspace. "
                    "Content:\nFree-tier RAG workspace for offline-safe retrieval, evaluation, and local context synthesis."
                ),
                node_id="readme#0",
                metadata={"file_name": "README.md"},
            )
        ]
        pipeline = LocalRAGPipeline(nodes=nodes)

        result = pipeline.chat_query("Me dá uma visão geral do projeto", strategy="bm25_only")

        self.assertEqual(result["intent"], "overview")
        self.assertIn("Free-tier RAG workspace", result["answer"])
        self.assertIn("Evidencia em: README.md", result["answer"])

    def test_chat_query_what_project_is_about_maps_to_overview(self):
        nodes = [
            pipeline_module.LocalTextNode(
                text="README section from README.md. Section: 03-advanced-rag. Content:\nFree-tier RAG workspace.",
                node_id="readme#0",
                metadata={"file_name": "README.md"},
            ),
            pipeline_module.LocalTextNode(
                text="README section from README.md. Section: Using your own GitHub projects. Content:\nClone repositories into data/raw.",
                node_id="readme#1",
                metadata={"file_name": "README.md"},
            ),
        ]
        pipeline = LocalRAGPipeline(nodes=nodes)

        result = pipeline.chat_query("what this project is about", strategy="hybrid_rerank")

        self.assertEqual(result["intent"], "overview")
        self.assertIn("Free-tier RAG workspace", result["answer"])
        self.assertNotIn("Clone repositories", result["answer"])

    def test_chat_query_low_evidence_refuses_to_invent(self):
        pipeline = LocalRAGPipeline(index=None, nodes=[], retriever=IrrelevantRetriever())

        result = pipeline.chat_query("Qual banco de dados o backend usa?", strategy="bm25_only")

        self.assertEqual(
            result["answer"],
            "Nao encontrei evidencia suficiente no contexto recuperado para responder sem inventar.",
        )
        self.assertLess(result["confidence"], 0.1)
        self.assertEqual(result["citations"][0]["source_doc"], "bananas.txt")

    def test_chat_query_summarizes_raw_json_context(self):
        raw_json = '{"name":"alpha","secrets":{"token":"abc123"},"items":[{"id":1},{"id":2}]}'
        retriever = SimpleNamespace(
            ablation_retrieve=lambda query, strategy: (
                [
                    NodeWithScore(
                        node=TextNode(
                            id_="json-1",
                            text=raw_json,
                            metadata={"file_name": "config.json"},
                        ),
                        score=0.8,
                    )
                ],
                {"strategy": strategy},
            )
        )
        pipeline = LocalRAGPipeline(index=None, nodes=[], retriever=retriever)

        result = pipeline.chat_query("What JSON fields exist?", strategy="bm25_only")

        self.assertIn("JSON document summary", result["answer"])
        self.assertIn("Top-level keys", result["answer"])
        self.assertNotIn('{"name"', result["answer"])
        self.assertNotIn('"token":"abc123"', result["answer"])
        self.assertIn("JSON document summary", result["citations"][0]["snippet"])


if __name__ == "__main__":
    unittest.main()
