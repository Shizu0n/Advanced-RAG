import json
import unittest
from unittest.mock import patch
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
            (raw_dir / "repo" / "node_modules").mkdir(parents=True)
            (raw_dir / "repo" / "node_modules" / "ignored.js").write_text("console.log('ignore')", encoding="utf-8")

            files = ingestion.load_or_download_sources(raw_dir=raw_dir)

        relative_paths = sorted(path.relative_to(raw_dir).as_posix() for path in files)

        self.assertEqual(relative_paths, ["repo/docs/notes.md", "repo/src/service.py"])

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

    def test_build_index_preserves_huggingface_metadata_from_previous_prepare_process(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            source_root = raw_dir / "Shizu0n-phi3-mini-sql-generator"
            source_root.mkdir(parents=True)
            (source_root / "README.md").write_text("# SQL generator\nThis model generates SQL queries.", encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
