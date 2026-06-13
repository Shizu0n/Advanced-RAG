import json
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

import ingestion
import source_loader


class IngestionTests(unittest.TestCase):
    def test_prepare_sources_is_available_from_ingestion_module(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            source_dir = base_dir / "repo"
            raw_dir = base_dir / "raw"
            source_dir.mkdir()
            (source_dir / "README.md").write_text("# Repo\n", encoding="utf-8")

            files = ingestion.prepare_sources([source_dir], raw_dir=raw_dir)

        self.assertEqual([path.name for path in files], ["README.md"])

    def test_load_or_download_sources_recurses_into_repo_directories(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            source_dir = raw_dir / "repo" / "src"
            source_dir.mkdir(parents=True)
            (source_dir / "service.py").write_text("def ping():\n    return 'pong'\n", encoding="utf-8")
            (raw_dir / "repo" / "docs").mkdir(parents=True)
            (raw_dir / "repo" / "docs" / "notes.md").write_text("# repo notes", encoding="utf-8")
            (raw_dir / "repo" / "docs" / "manual.pdf").write_bytes(b"%PDF-1.4 fixture")
            (raw_dir / "repo" / "docs" / "brief.docx").write_bytes(b"docx fixture")
            (raw_dir / "repo" / "node_modules").mkdir(parents=True)
            (raw_dir / "repo" / "node_modules" / "ignored.js").write_text("console.log('ignore')", encoding="utf-8")

            files = ingestion.load_or_download_sources(raw_dir=raw_dir)

        relative_paths = sorted(path.relative_to(raw_dir).as_posix() for path in files)

        self.assertEqual(
            relative_paths,
            ["repo/docs/brief.docx", "repo/docs/manual.pdf", "repo/docs/notes.md", "repo/src/service.py"],
        )

    def test_safe_read_text_extracts_pdf_pages(self):
        class FakePage:
            def __init__(self, text):
                self._text = text

            def extract_text(self):
                return self._text

        class FakePdfReader:
            def __init__(self, path):
                self.path = path
                self.pages = [FakePage("First page"), FakePage(""), FakePage("Second page")]

        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "manual.pdf"
            source.write_bytes(b"%PDF-1.4 fixture")
            fake_module = SimpleNamespace(PdfReader=FakePdfReader)

            with patch.dict(sys.modules, {"PyPDF2": fake_module}):
                text = ingestion._safe_read_text(source)

        self.assertEqual(text, "First page\n\nSecond page")

    def test_safe_read_text_extracts_docx_paragraphs_and_tables(self):
        fake_document = SimpleNamespace(
            paragraphs=[SimpleNamespace(text="Intro paragraph"), SimpleNamespace(text="")],
            tables=[
                SimpleNamespace(
                    rows=[
                        SimpleNamespace(
                            cells=[SimpleNamespace(text="Name"), SimpleNamespace(text="Value")]
                        )
                    ]
                )
            ],
        )
        fake_module = SimpleNamespace(Document=lambda path: fake_document)

        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "brief.docx"
            source.write_bytes(b"docx fixture")

            with patch.dict(sys.modules, {"docx": fake_module}):
                text = ingestion._safe_read_text(source)

        self.assertEqual(text, "Intro paragraph\nName\tValue")

    def test_run_chunking_ablation_writes_rows_without_rebuilding_index(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()
            source = raw_dir / "notes.md"
            source.write_text(("Alpha beta gamma delta. " * 80).strip(), encoding="utf-8")
            output_path = Path(tmpdir) / "chunking.csv"

            rows = ingestion.run_chunking_ablation(
                source_files=[source],
                chunk_sizes=[128, 256],
                chunk_overlap_tokens=16,
                output_path=output_path,
            )

            self.assertEqual([row["chunk_size"] for row in rows], [128, 256])
            self.assertTrue(all(row["file_count"] == 1 for row in rows))
            self.assertTrue(all(row["chunk_count"] >= 1 for row in rows))
            self.assertTrue(output_path.exists())

    def test_embedding_model_comparison_skips_when_model_downloads_disabled(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()
            source = raw_dir / "notes.md"
            source.write_text("Embedding comparison sample text.", encoding="utf-8")

            rows = ingestion.run_embedding_model_comparison(
                source_files=[source],
                models=["model-a", "model-b"],
                allow_model_downloads=False,
            )

        self.assertEqual([row["status"] for row in rows], ["skipped_model_downloads_disabled", "skipped_model_downloads_disabled"])
        self.assertTrue(all(row["embedding_dim"] == 0 for row in rows))

    def test_embedding_model_comparison_embeds_sample_when_allowed(self):
        class FakeEmbedding:
            def __init__(self, model_name):
                self.model_name = model_name

            def get_text_embedding_batch(self, samples):
                return [[0.1, 0.2, 0.3] for _ in samples]

        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()
            source = raw_dir / "notes.md"
            source.write_text("Embedding comparison sample text.", encoding="utf-8")

            with patch.object(ingestion, "HuggingFaceEmbedding", FakeEmbedding):
                rows = ingestion.run_embedding_model_comparison(
                    source_files=[source],
                    models=["model-a"],
                    sample_size=1,
                    allow_model_downloads=True,
                )

        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["model"], "model-a")
        self.assertEqual(rows[0]["sample_count"], 1)
        self.assertEqual(rows[0]["embedding_dim"], 3)

    def test_load_or_download_sources_requires_network_opt_in_when_raw_empty(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"

            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "ALLOW_DOCS_DOWNLOAD"):
                    ingestion.load_or_download_sources(raw_dir=raw_dir)

            with (
                patch.dict("os.environ", {"ALLOW_DOCS_DOWNLOAD": "1"}, clear=True),
                patch.object(ingestion, "download_python_tutorial_pages", return_value=[raw_dir / "doc.txt"]) as download,
            ):
                files = ingestion.load_or_download_sources(raw_dir=raw_dir)

            self.assertEqual(files, [raw_dir / "doc.txt"])
            download.assert_called_once()


class CurrentSourceTests(unittest.TestCase):
    def test_write_current_source_creates_correct_json(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "my-repo"
            raw_dir.mkdir(parents=True)
            source_path = Path(tmpdir) / "current_source.json"

            with patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path):
                ingestion.write_current_source(raw_dir=raw_dir, file_count=5, chunk_count=42, source_input="https://example.com/repo")

            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["source_input"], "https://example.com/repo")
            self.assertEqual(data["source_type"], "local")
            self.assertEqual(data["source_slug"], "my-repo")
            self.assertEqual(data["file_count"], 5)
            self.assertEqual(data["chunk_count"], 42)
            self.assertIn("indexed_at", data)

    def test_write_current_source_detects_huggingface_type(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "huggingface-meta-llama"
            raw_dir.mkdir(parents=True)
            source_path = Path(tmpdir) / "current_source.json"

            with patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path):
                ingestion.write_current_source(raw_dir=raw_dir, file_count=1, chunk_count=10)

            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["source_type"], "huggingface")
            self.assertEqual(data["source_slug"], "huggingface-meta-llama")

    def test_write_current_source_detects_github_type(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "github-owner-repo"
            raw_dir.mkdir(parents=True)
            source_path = Path(tmpdir) / "current_source.json"

            with patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path):
                ingestion.write_current_source(raw_dir=raw_dir, file_count=3, chunk_count=20)

            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["source_type"], "github")
            self.assertEqual(data["source_slug"], "github-owner-repo")

    def test_write_current_source_defaults_to_local_type(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "my-project"
            raw_dir.mkdir(parents=True)
            source_path = Path(tmpdir) / "current_source.json"

            with patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path):
                ingestion.write_current_source(raw_dir=raw_dir, file_count=2, chunk_count=15)

            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["source_type"], "local")

    def test_load_current_source_reads_json(self):
        with TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "current_source.json"
            expected = {"source_slug": "test-repo", "source_type": "local", "file_count": 3}
            source_path.write_text(json.dumps(expected), encoding="utf-8")

            with patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path):
                result = ingestion.load_current_source()

            self.assertEqual(result["source_slug"], "test-repo")
            self.assertEqual(result["source_type"], "local")
            self.assertEqual(result["file_count"], 3)

    def test_load_current_source_returns_none_when_missing(self):
        with TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "nonexistent.json"

            with patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path):
                result = ingestion.load_current_source()

            self.assertIsNone(result)

    def test_default_paths_are_project_local_when_cwd_changes(self):
        old_cwd = os.getcwd()
        with TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                self.assertEqual(ingestion.RAW_DIR, ingestion.PROJECT_ROOT / "data" / "raw")
                self.assertEqual(ingestion.CHROMA_DIR, ingestion.PROJECT_ROOT / "chroma_db")
                self.assertEqual(ingestion.CURRENT_SOURCE_PATH, ingestion.PROJECT_ROOT / "data" / "current_source.json")
            finally:
                os.chdir(old_cwd)

    def test_build_index_failure_clears_stale_indexed_source_metadata_and_collection(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "new-source"
            raw_dir.mkdir(parents=True)
            (raw_dir / "README.md").write_text("# New source\n", encoding="utf-8")
            source_path = Path(tmpdir) / "current_source.json"
            chroma_dir = Path(tmpdir) / "chroma_db"
            source_path.write_text('{"source_slug":"old-source"}', encoding="utf-8")
            chroma_dir.mkdir()

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding", side_effect=RuntimeError("embedding failed")),
                patch.object(ingestion, "chromadb") as mock_chromadb,
            ):
                with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                    ingestion.build_index(raw_dir=raw_dir)

            self.assertFalse(source_path.exists())
            self.assertTrue(chroma_dir.exists())
            self.assertEqual(mock_chromadb.PersistentClient.return_value.delete_collection.call_count, 2)
            mock_chromadb.PersistentClient.return_value.delete_collection.assert_any_call(ingestion.CHROMA_COLLECTION_NAME)

    def test_clear_indexed_source_artifacts_removes_chroma_dir_when_client_cannot_open_it(self):
        with TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "current_source.json"
            chroma_dir = Path(tmpdir) / "chroma_db"
            source_path.write_text('{"source_slug":"old-source"}', encoding="utf-8")
            chroma_dir.mkdir()
            (chroma_dir / "chroma.sqlite3").write_text("stale", encoding="utf-8")

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "chromadb") as mock_chromadb,
            ):
                mock_chromadb.PersistentClient.side_effect = RuntimeError("cannot open chroma")
                ingestion.clear_indexed_source_artifacts()

            self.assertFalse(source_path.exists())
            self.assertTrue(chroma_dir.exists())
            self.assertEqual(list(chroma_dir.iterdir()), [])

    def test_build_index_retries_chroma_client_after_reset_when_existing_dir_is_invalid(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "test-repo"
            raw_dir.mkdir(parents=True)
            (raw_dir / "readme.md").write_text("# Hello\nThis is a test document with real content.", encoding="utf-8")
            source_path = Path(tmpdir) / "current_source.json"
            chroma_dir = Path(tmpdir) / "chroma_db"
            chroma_dir.mkdir()
            (chroma_dir / "chroma.sqlite3").write_text("stale", encoding="utf-8")

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb") as mock_chromadb,
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_client = MagicMock()
                mock_chromadb.PersistentClient.side_effect = [RuntimeError("bad chroma"), mock_client]
                mock_index.return_value = "fake_index"
                ingestion.build_index(raw_dir=raw_dir)

            self.assertTrue(source_path.exists())
            self.assertEqual(mock_chromadb.PersistentClient.call_count, 2)
            mock_client.create_collection.assert_called_once_with(ingestion.CHROMA_COLLECTION_NAME)

    def test_build_index_retries_chroma_client_when_delete_collection_finds_invalid_db(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "test-repo"
            raw_dir.mkdir(parents=True)
            (raw_dir / "readme.md").write_text("# Hello\nThis is a test document with real content.", encoding="utf-8")
            source_path = Path(tmpdir) / "current_source.json"
            chroma_dir = Path(tmpdir) / "chroma_db"
            chroma_dir.mkdir()
            (chroma_dir / "chroma.sqlite3").write_text("stale", encoding="utf-8")

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb") as mock_chromadb,
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                bad_client = MagicMock()
                bad_client.delete_collection.side_effect = RuntimeError("no such table: tenants")
                good_client = MagicMock()
                mock_chromadb.PersistentClient.side_effect = [bad_client, good_client]
                mock_index.return_value = "fake_index"
                ingestion.build_index(raw_dir=raw_dir)

            self.assertTrue(source_path.exists())
            self.assertEqual(mock_chromadb.PersistentClient.call_count, 2)
            good_client.create_collection.assert_called_once_with(ingestion.CHROMA_COLLECTION_NAME)

    def test_build_index_retries_whole_chroma_write_when_vector_index_finds_invalid_db(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "test-repo"
            raw_dir.mkdir(parents=True)
            (raw_dir / "readme.md").write_text("# Hello\nThis is a test document with real content.", encoding="utf-8")
            source_path = Path(tmpdir) / "current_source.json"
            chroma_dir = Path(tmpdir) / "chroma_db"
            chroma_dir.mkdir()
            (chroma_dir / "chroma.sqlite3").write_text("stale", encoding="utf-8")

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb") as mock_chromadb,
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                first_client = MagicMock()
                second_client = MagicMock()
                mock_chromadb.PersistentClient.side_effect = [first_client, second_client]
                mock_index.side_effect = [RuntimeError("no such table: tenants"), "fake_index"]
                ingestion.build_index(raw_dir=raw_dir)

            self.assertTrue(source_path.exists())
            self.assertEqual(mock_chromadb.PersistentClient.call_count, 2)
            self.assertEqual(mock_index.call_count, 2)
            second_client.create_collection.assert_called_once_with(ingestion.CHROMA_COLLECTION_NAME)

    def test_build_index_writes_current_source_json(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "test-repo"
            raw_dir.mkdir(parents=True)
            (raw_dir / "readme.md").write_text("# Hello\nThis is a test document with real content.", encoding="utf-8")
            source_path = Path(tmpdir) / "current_source.json"
            chroma_dir = Path(tmpdir) / "chroma_db"

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding") as mock_embed,
                patch.object(ingestion, "chromadb"),
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_index.return_value = "fake_index"
                index, nodes = ingestion.build_index(raw_dir=raw_dir)

            self.assertTrue(source_path.exists())
            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["source_slug"], "test-repo")
            self.assertEqual(data["source_type"], "local")
            self.assertGreaterEqual(data["file_count"], 1)
            self.assertGreaterEqual(data["chunk_count"], 1)

    def test_build_index_records_prepared_huggingface_source_metadata(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            source_path = Path(tmpdir) / "current_source.json"
            chroma_dir = Path(tmpdir) / "chroma_db"

            with patch.object(source_loader, "_fetch_huggingface_card") as fetch_card:
                fetch_card.return_value = Path(tmpdir) / "README.md"
                fetch_card.return_value.write_text("# SQL generator\nThis model generates SQL queries.", encoding="utf-8")
                files = ingestion.prepare_sources(
                    ["hf:Shizu0n/phi3-mini-sql-generator"],
                    raw_dir=raw_dir,
                    allow_huggingface_fetch=True,
                )

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb"),
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_index.return_value = "fake_index"
                ingestion.build_index(source_files=files, raw_dir=raw_dir)

            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["source_type"], "huggingface")
            self.assertEqual(data["source_input"], "hf:Shizu0n/phi3-mini-sql-generator")
            self.assertEqual(data["source_slug"], "shizu0n-phi3-mini-sql-generator")

    def test_build_index_records_prepared_github_source_metadata_after_stale_hf_source(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            raw_dir = base_dir / "raw"
            source_path = base_dir / "current_source.json"
            source_path.write_text(
                json.dumps(
                    {
                        "source_input": "hf:Shizu0n/phi3-mini-sql-generator",
                        "source_type": "huggingface",
                        "source_slug": "shizu0n-phi3-mini-sql-generator",
                        "indexed_at": "2026-05-20T00:00:00+00:00",
                        "file_count": 1,
                        "chunk_count": 10,
                    }
                ),
                encoding="utf-8",
            )
            chroma_dir = base_dir / "chroma_db"
            fetched_dir = base_dir / "downloaded"
            fetched_dir.mkdir()
            (fetched_dir / "README.md").write_text("# Repo\nGitHub fixture content.", encoding="utf-8")

            with patch.object(source_loader, "_fetch_github_repository", return_value=fetched_dir):
                files = ingestion.prepare_sources(
                    ["https://github.com/acme/repo"],
                    raw_dir=raw_dir,
                    allow_github_fetch=True,
                    clear_existing=True,
                )

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb"),
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_index.return_value = "fake_index"
                ingestion.build_index(source_files=files, raw_dir=raw_dir)

            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["source_type"], "github")
            self.assertEqual(data["source_input"], "https://github.com/acme/repo")
            self.assertEqual(data["source_slug"], "acme-repo")
            self.assertEqual(data["file_count"], 1)
            self.assertIsNotNone(data["indexed_at"])

    def test_build_index_records_prepared_local_source_metadata_after_stale_hf_source(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            project_dir = base_dir / "project"
            raw_dir = base_dir / "raw"
            source_path = base_dir / "current_source.json"
            chroma_dir = base_dir / "chroma_db"
            project_dir.mkdir()
            (project_dir / "README.md").write_text("# Local\nLocal fixture content.", encoding="utf-8")
            source_path.write_text(
                json.dumps(
                    {
                        "source_input": "hf:Shizu0n/phi3-mini-sql-generator",
                        "source_type": "huggingface",
                        "source_slug": "shizu0n-phi3-mini-sql-generator",
                        "indexed_at": "2026-05-20T00:00:00+00:00",
                        "file_count": 1,
                        "chunk_count": 10,
                    }
                ),
                encoding="utf-8",
            )

            files = ingestion.prepare_sources([project_dir], raw_dir=raw_dir, clear_existing=True)

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb"),
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_index.return_value = "fake_index"
                ingestion.build_index(source_files=files, raw_dir=raw_dir)

            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["source_type"], "local")
            self.assertEqual(data["source_input"], str(project_dir))
            self.assertTrue(data["source_slug"].startswith("project-"))
            self.assertEqual(data["file_count"], 1)
            self.assertIsNotNone(data["indexed_at"])

    def test_build_index_with_explicit_files_without_prepared_metadata_does_not_reuse_stale_current_source(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            raw_dir = base_dir / "raw"
            source_root = raw_dir / "new-source"
            source_root.mkdir(parents=True)
            source_file = source_root / "README.md"
            source_file.write_text("# New Source\nFresh explicit source content.", encoding="utf-8")
            source_path = base_dir / "current_source.json"
            chroma_dir = base_dir / "chroma_db"
            source_path.write_text(
                json.dumps(
                    {
                        "source_input": "hf:Shizu0n/phi3-mini-sql-generator",
                        "source_type": "huggingface",
                        "source_slug": "shizu0n-phi3-mini-sql-generator",
                        "indexed_at": "2026-05-20T00:00:00+00:00",
                        "file_count": 1,
                        "chunk_count": 10,
                    }
                ),
                encoding="utf-8",
            )
            source_loader.PREPARED_SOURCE_METADATA.clear()

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb"),
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_index.return_value = "fake_index"
                ingestion.build_index(source_files=[source_file], raw_dir=raw_dir)

            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["source_type"], "local")
            self.assertEqual(data["source_input"], "")
            self.assertEqual(data["source_slug"], "new-source")

    def test_build_index_preserves_huggingface_metadata_from_durable_prepared_source_after_restart(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            source_root = raw_dir / "Shizu0n-phi3-mini-sql-generator"
            source_root.mkdir(parents=True)
            (source_root / "README.md").write_text("# SQL generator\nThis model generates SQL queries.", encoding="utf-8")
            source_path = Path(tmpdir) / "current_source.json"
            prepared_path = Path(tmpdir) / "prepared_source.json"
            prepared_path.write_text(
                json.dumps(
                    {
                        "source_input": "hf:Shizu0n/phi3-mini-sql-generator",
                        "source_type": "huggingface",
                        "source_slug": "shizu0n-phi3-mini-sql-generator",
                        "indexed_at": None,
                        "file_count": 1,
                        "chunk_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            chroma_dir = Path(tmpdir) / "chroma_db"
            source_loader.PREPARED_SOURCE_METADATA.clear()

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(source_loader, "PREPARED_SOURCE_PATH", prepared_path),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb"),
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_index.return_value = "fake_index"
                ingestion.build_index(raw_dir=raw_dir)

            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["source_type"], "huggingface")
            self.assertEqual(data["source_input"], "hf:Shizu0n/phi3-mini-sql-generator")
            self.assertEqual(data["source_slug"], "shizu0n-phi3-mini-sql-generator")
            self.assertIsNotNone(data["indexed_at"])
            self.assertGreaterEqual(data["chunk_count"], 1)

    def test_build_index_scopes_to_prepared_current_source_when_raw_has_stale_sources(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            source_root = raw_dir / "Shizu0n-phi3-mini-sql-generator"
            source_root.mkdir(parents=True)
            (source_root / "README.md").write_text("# SQL generator\nThis model generates SQL queries.", encoding="utf-8")
            stale_root = raw_dir / "old-source"
            stale_root.mkdir(parents=True)
            (stale_root / "README.md").write_text("# Old source\nThis should not be indexed.", encoding="utf-8")
            source_path = Path(tmpdir) / "current_source.json"
            source_path.write_text(
                json.dumps(
                    {
                        "source_input": "hf:Shizu0n/phi3-mini-sql-generator",
                        "source_type": "huggingface",
                        "source_slug": "shizu0n-phi3-mini-sql-generator",
                        "indexed_at": None,
                        "file_count": 1,
                        "chunk_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            chroma_dir = Path(tmpdir) / "chroma_db"

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb"),
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_index.return_value = "fake_index"
                ingestion.build_index(raw_dir=raw_dir)

            data = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(data["file_count"], 1)

    def test_build_index_replaces_existing_chroma_collection(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "test-repo"
            raw_dir.mkdir(parents=True)
            (raw_dir / "readme.md").write_text("# Hello\nThis is a test document with real content.", encoding="utf-8")
            source_path = Path(tmpdir) / "current_source.json"
            chroma_dir = Path(tmpdir) / "chroma_db"

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb") as mock_chromadb,
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_index.return_value = "fake_index"
                ingestion.build_index(raw_dir=raw_dir)

            client = mock_chromadb.PersistentClient.return_value
            client.delete_collection.assert_called_once_with(ingestion.CHROMA_COLLECTION_NAME)
            client.create_collection.assert_called_once_with(ingestion.CHROMA_COLLECTION_NAME)
            client.get_or_create_collection.assert_not_called()

    def test_build_index_allows_first_chroma_collection_creation(self):
        not_found_error = ingestion.chromadb.errors.NotFoundError
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "test-repo"
            raw_dir.mkdir(parents=True)
            (raw_dir / "readme.md").write_text("# Hello\nThis is a test document with real content.", encoding="utf-8")
            source_path = Path(tmpdir) / "current_source.json"
            chroma_dir = Path(tmpdir) / "chroma_db"

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb") as mock_chromadb,
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_chromadb.errors.NotFoundError = not_found_error
                mock_chromadb.PersistentClient.return_value.delete_collection.side_effect = not_found_error("missing")
                mock_index.return_value = "fake_index"
                ingestion.build_index(raw_dir=raw_dir)

            client = mock_chromadb.PersistentClient.return_value
            client.create_collection.assert_called_once_with(ingestion.CHROMA_COLLECTION_NAME)

    def test_build_index_allows_chroma_value_error_for_missing_collection(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "test-repo"
            raw_dir.mkdir(parents=True)
            (raw_dir / "readme.md").write_text("# Hello\nThis is a test document with real content.", encoding="utf-8")
            source_path = Path(tmpdir) / "current_source.json"
            chroma_dir = Path(tmpdir) / "chroma_db"

            with (
                patch.object(ingestion, "CURRENT_SOURCE_PATH", source_path),
                patch.object(ingestion, "CHROMA_DIR", chroma_dir),
                patch.object(ingestion, "HuggingFaceEmbedding"),
                patch.object(ingestion, "chromadb") as mock_chromadb,
                patch.object(ingestion, "VectorStoreIndex") as mock_index,
            ):
                mock_chromadb.PersistentClient.return_value.delete_collection.side_effect = ValueError(
                    "Collection advanced_rag does not exist."
                )
                mock_index.return_value = "fake_index"
                ingestion.build_index(raw_dir=raw_dir)

            client = mock_chromadb.PersistentClient.return_value
            client.create_collection.assert_called_once_with(ingestion.CHROMA_COLLECTION_NAME)


if __name__ == "__main__":
    unittest.main()
