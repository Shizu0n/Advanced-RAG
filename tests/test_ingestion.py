import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import ingestion


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


if __name__ == "__main__":
    unittest.main()
