import json
import os
import zipfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

import source_loader


class SourceLoaderTests(unittest.TestCase):
    def setUp(self):
        source_loader.PREPARED_SOURCE_METADATA.clear()

    def test_prepare_sources_copies_supported_files_from_local_directory(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            repo_dir = base_dir / "sample-repo"
            raw_dir = base_dir / "raw"
            (repo_dir / "src").mkdir(parents=True)
            (repo_dir / "docs").mkdir()
            (repo_dir / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (repo_dir / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
            (repo_dir / "src" / "image.png").write_bytes(b"not text")

            files = source_loader.prepare_sources([repo_dir], raw_dir=raw_dir)

            relative_paths = sorted(path.relative_to(raw_dir).as_posix() for path in files)
            target_root = Path(relative_paths[0]).parts[0]

            self.assertRegex(target_root, r"^sample-repo-[0-9a-f]{12}$")
            self.assertEqual(
                relative_paths,
                [f"{target_root}/docs/guide.md", f"{target_root}/src/app.py"],
            )
            self.assertEqual((raw_dir / target_root / "src" / "app.py").read_text(encoding="utf-8"), "print('ok')\n")

    def test_prepare_sources_ignores_generated_and_private_directories(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            repo_dir = base_dir / "repo"
            raw_dir = base_dir / "raw"
            (repo_dir / "src").mkdir(parents=True)
            (repo_dir / ".git").mkdir()
            (repo_dir / "node_modules").mkdir()
            (repo_dir / "__pycache__").mkdir()
            (repo_dir / "src" / "main.ts").write_text("export const ok = true;\n", encoding="utf-8")
            (repo_dir / "package-lock.json").write_text('{"name":"ignored"}\n', encoding="utf-8")
            (repo_dir / "pnpm-lock.yaml").write_text("ignored: true\n", encoding="utf-8")
            (repo_dir / ".git" / "config").write_text("[core]\n", encoding="utf-8")
            (repo_dir / "node_modules" / "pkg.js").write_text("ignored\n", encoding="utf-8")
            (repo_dir / "__pycache__" / "cache.py").write_text("ignored\n", encoding="utf-8")

            files = source_loader.prepare_sources([repo_dir], raw_dir=raw_dir)

            relative_paths = sorted(path.relative_to(raw_dir).as_posix() for path in files)
            target_root = Path(relative_paths[0]).parts[0]

            self.assertEqual(relative_paths, [f"{target_root}/src/main.ts"])

    def test_prepare_sources_skips_raw_dir_when_it_is_inside_source(self):
        with TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            raw_dir = project_dir / "data" / "raw"
            (project_dir / "src").mkdir(parents=True)
            (raw_dir / "existing-output").mkdir(parents=True)
            (project_dir / "src" / "app.py").write_text("print('source')\n", encoding="utf-8")
            (raw_dir / "existing-output" / "old.md").write_text("# already prepared\n", encoding="utf-8")

            files = source_loader.prepare_sources([project_dir], raw_dir=raw_dir)

            relative_paths = sorted(path.relative_to(raw_dir).as_posix() for path in files)

            self.assertEqual(len(relative_paths), 1)
            self.assertTrue(relative_paths[0].endswith("/src/app.py"))
            self.assertNotIn("existing-output/old.md", relative_paths)

    def test_prepare_sources_keeps_same_basename_sources_separate(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            first = base_dir / "one" / "repo"
            second = base_dir / "two" / "repo"
            raw_dir = base_dir / "raw"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "README.md").write_text("# first\n", encoding="utf-8")
            (second / "README.md").write_text("# second\n", encoding="utf-8")

            files = source_loader.prepare_sources([first, second], raw_dir=raw_dir)

            relative_paths = sorted(path.relative_to(raw_dir).as_posix() for path in files)
            target_roots = sorted({Path(path).parts[0] for path in relative_paths})

            self.assertEqual(len(target_roots), 2)
            self.assertTrue(all(root.startswith("repo-") for root in target_roots))
            self.assertEqual([Path(path).parts[-1] for path in relative_paths], ["README.md", "README.md"])

    def test_prepare_sources_does_not_follow_symlinked_files(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            outside = base_dir / "outside"
            repo_dir = base_dir / "repo"
            raw_dir = base_dir / "raw"
            outside.mkdir()
            repo_dir.mkdir()
            (outside / "secret.py").write_text("SECRET = True\n", encoding="utf-8")
            link = repo_dir / "secret.py"

            try:
                link.symlink_to(outside / "secret.py")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable in this environment")

            files = source_loader.prepare_sources([repo_dir], raw_dir=raw_dir)

            self.assertEqual(files, [])

    def test_prepare_sources_clear_existing_removes_stale_raw_subtree_before_local_copy(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            repo_dir = base_dir / "repo"
            raw_dir = base_dir / "raw"
            (repo_dir / "src").mkdir(parents=True)
            (repo_dir / "src" / "main.py").write_text("print('fresh')\n", encoding="utf-8")
            stale_dir = raw_dir / "stale-source"
            stale_dir.mkdir(parents=True)
            (stale_dir / "old.md").write_text("# stale\n", encoding="utf-8")
            source_loader.PREPARED_SOURCE_METADATA[(stale_dir / "old.md").resolve().as_posix()] = {
                "source_input": "stale",
                "source_type": "local",
                "source_slug": "stale-source",
                "indexed_at": None,
                "file_count": 1,
                "chunk_count": 0,
            }

            files = source_loader.prepare_sources([repo_dir], raw_dir=raw_dir, clear_existing=True)

            self.assertEqual(len(files), 1)
            self.assertFalse(stale_dir.exists())
            self.assertEqual(len(source_loader.PREPARED_SOURCE_METADATA), 1)
            only_path = next(iter(source_loader.PREPARED_SOURCE_METADATA))
            self.assertTrue(only_path.endswith("/src/main.py"))

    def test_prepare_sources_clear_existing_preserves_raw_dir_when_root_cannot_be_removed(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            repo_dir = base_dir / "repo"
            raw_dir = base_dir / "raw"
            (repo_dir / "src").mkdir(parents=True)
            (repo_dir / "src" / "main.py").write_text("print('fresh')\n", encoding="utf-8")
            stale_dir = raw_dir / "stale-source"
            stale_dir.mkdir(parents=True)
            (stale_dir / "old.md").write_text("# stale\n", encoding="utf-8")
            original_rmtree = source_loader.shutil.rmtree

            def rmtree_unless_raw_root(path):
                if Path(path) == raw_dir:
                    raise PermissionError("raw root is locked")
                original_rmtree(path)

            with patch.object(source_loader.shutil, "rmtree", side_effect=rmtree_unless_raw_root):
                files = source_loader.prepare_sources([repo_dir], raw_dir=raw_dir, clear_existing=True)

            self.assertTrue(raw_dir.exists())
            self.assertEqual(len(files), 1)
            self.assertFalse(stale_dir.exists())

    def test_prepare_sources_persists_pending_metadata_separately_from_current_source(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            repo_dir = base_dir / "repo"
            raw_dir = base_dir / "raw"
            prepared_path = base_dir / "prepared_source.json"
            current_path = base_dir / "current_source.json"
            repo_dir.mkdir()
            (repo_dir / "README.md").write_text("# Fresh repo\n", encoding="utf-8")

            with (
                patch.object(source_loader, "PREPARED_SOURCE_PATH", prepared_path),
                patch.object(source_loader, "CURRENT_SOURCE_PATH", current_path),
            ):
                source_loader.prepare_sources([repo_dir], raw_dir=raw_dir)

            self.assertTrue(prepared_path.exists())
            self.assertFalse(current_path.exists())
            metadata = json.loads(prepared_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_type"], "local")
            self.assertEqual(metadata["source_input"], str(repo_dir))
            self.assertTrue(metadata["source_slug"].startswith("repo-"))
            self.assertIsNone(metadata["indexed_at"])

    def test_prepare_sources_default_raw_dir_is_project_local(self):
        self.assertEqual(source_loader.prepare_sources.__defaults__[0], source_loader.PROJECT_ROOT / "data" / "raw")

    def test_prepare_sources_registers_metadata_for_local_directory_files(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            project_dir = base_dir / "My Project"
            raw_dir = base_dir / "raw"
            (project_dir / "src").mkdir(parents=True)
            (project_dir / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (project_dir / "README.md").write_text("# Test\n", encoding="utf-8")

            files = source_loader.prepare_sources([project_dir], raw_dir=raw_dir)

            metadata_by_path = source_loader.PREPARED_SOURCE_METADATA
            self.assertEqual(set(metadata_by_path), {path.resolve().as_posix() for path in files})
            slug = source_loader._local_target_name(project_dir).lower()
            for path in files:
                metadata = metadata_by_path[path.resolve().as_posix()]
                self.assertEqual(metadata["source_input"], str(project_dir))
                self.assertEqual(metadata["source_type"], "local")
                self.assertEqual(metadata["source_slug"], slug)
                self.assertIsNone(metadata["indexed_at"])
                self.assertEqual(metadata["file_count"], 2)
                self.assertEqual(metadata["chunk_count"], 0)

    def test_prepare_sources_registers_metadata_for_nested_local_file_path(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            source_file = base_dir / "nested space" / "pkg" / "module.py"
            raw_dir = base_dir / "raw"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("print('nested')\n", encoding="utf-8")

            files = source_loader.prepare_sources([source_file], raw_dir=raw_dir)

            self.assertEqual(len(files), 1)
            metadata = source_loader.PREPARED_SOURCE_METADATA[files[0].resolve().as_posix()]
            self.assertEqual(metadata["source_input"], str(source_file))
            self.assertEqual(metadata["source_type"], "local")
            self.assertEqual(metadata["source_slug"], source_loader._local_target_name(source_file).lower())
            self.assertIsNone(metadata["indexed_at"])
            self.assertEqual(metadata["file_count"], 1)
            self.assertEqual(metadata["chunk_count"], 0)

    def test_prepare_sources_registers_metadata_for_github_files(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            fetched_dir = base_dir / "downloaded" / "project-main"
            raw_dir = base_dir / "raw"
            (fetched_dir / "src").mkdir(parents=True)
            (fetched_dir / "README.md").write_text("# fetched\n", encoding="utf-8")
            (fetched_dir / "src" / "app.py").write_text("print('fetched')\n", encoding="utf-8")

            with patch.object(source_loader, "_fetch_github_repository", return_value=fetched_dir):
                files = source_loader.prepare_sources(
                    ["https://github.com/Example/Project"],
                    raw_dir=raw_dir,
                    allow_github_fetch=True,
                )

            metadata_by_path = source_loader.PREPARED_SOURCE_METADATA
            self.assertEqual(set(metadata_by_path), {path.resolve().as_posix() for path in files})
            for path in files:
                metadata = metadata_by_path[path.resolve().as_posix()]
                self.assertEqual(metadata["source_input"], "https://github.com/Example/Project")
                self.assertEqual(metadata["source_type"], "github")
                self.assertEqual(metadata["source_slug"], source_loader._github_target_name("Example", "Project").lower())
                self.assertIsNone(metadata["indexed_at"])
                self.assertEqual(metadata["file_count"], 2)
                self.assertEqual(metadata["chunk_count"], 0)

    def test_supported_file_rejects_symlinks(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "target.py"
            link = base_dir / "link.py"
            target.write_text("print('secret')\n", encoding="utf-8")
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable in this environment")

            self.assertFalse(source_loader._is_supported_file(link))

    def test_prepare_sources_rejects_github_url_without_opt_in(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"

            with self.assertRaisesRegex(RuntimeError, "allow_github_fetch"):
                source_loader.prepare_sources(
                    ["https://github.com/example/project"],
                    raw_dir=raw_dir,
                    allow_github_fetch=False,
                )

    def test_prepare_sources_uses_mocked_github_fetch_when_opted_in(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            fetched_dir = base_dir / "downloaded"
            raw_dir = base_dir / "raw"
            (fetched_dir / "project-main").mkdir(parents=True)
            (fetched_dir / "project-main" / "README.md").write_text("# fetched\n", encoding="utf-8")

            with patch.object(source_loader, "_fetch_github_repository", return_value=fetched_dir / "project-main") as fetch:
                files = source_loader.prepare_sources(
                    ["https://github.com/example/project"],
                    raw_dir=raw_dir,
                    allow_github_fetch=True,
                )

            fetch.assert_called_once()
            self.assertEqual([path.relative_to(raw_dir).as_posix() for path in files], ["example-project/README.md"])

    def test_prepare_sources_keeps_same_repo_names_separate_by_owner(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            raw_dir = base_dir / "raw"
            first_fetch = base_dir / "downloads" / "first"
            second_fetch = base_dir / "downloads" / "second"
            first_fetch.mkdir(parents=True)
            second_fetch.mkdir(parents=True)
            (first_fetch / "README.md").write_text("# first\n", encoding="utf-8")
            (second_fetch / "README.md").write_text("# second\n", encoding="utf-8")

            def fake_fetch(source, download_dir):
                if "first" in source:
                    return first_fetch
                return second_fetch

            with patch.object(source_loader, "_fetch_github_repository", side_effect=fake_fetch):
                files = source_loader.prepare_sources(
                    [
                        "https://github.com/first/project",
                        "https://github.com/second/project",
                    ],
                    raw_dir=raw_dir,
                    allow_github_fetch=True,
                )

            self.assertEqual(
                sorted(path.relative_to(raw_dir).as_posix() for path in files),
                ["first-project/README.md", "second-project/README.md"],
            )

    def test_prepare_sources_allows_github_fetch_from_env(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            fetched_dir = base_dir / "downloaded" / "repo-main"
            raw_dir = base_dir / "raw"
            fetched_dir.mkdir(parents=True)
            (fetched_dir / "README.md").write_text("# env\n", encoding="utf-8")

            with patch.dict(os.environ, {"ALLOW_GITHUB_FETCH": "1"}):
                with patch.object(source_loader, "_fetch_github_repository", return_value=fetched_dir):
                    files = source_loader.prepare_sources(
                        ["https://github.com/acme/repo"],
                        raw_dir=raw_dir,
                    )

            self.assertEqual([path.relative_to(raw_dir).as_posix() for path in files], ["acme-repo/README.md"])

    def test_fetch_github_repository_rejects_zip_over_size_limit(self):
        with TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir)

            class FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self, size=-1):
                    return b"x" * 8

            with patch.object(source_loader, "urlopen", return_value=FakeResponse()):
                with self.assertRaisesRegex(RuntimeError, "exceeds maximum"):
                    source_loader._fetch_github_repository(
                        "https://github.com/acme/repo",
                        download_dir,
                        max_zip_bytes=4,
                    )

    def test_safe_zip_extraction_skips_unsupported_ignored_and_unsafe_paths(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            zip_path = base_dir / "repo.zip"
            extract_dir = base_dir / "extract"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("repo-main/README.md", "# ok\n")
                archive.writestr("repo-main/node_modules/pkg/index.js", "ignored\n")
                archive.writestr("repo-main/assets/logo.png", "ignored\n")
                archive.writestr("../escape.md", "bad\n")

            with self.assertRaisesRegex(RuntimeError, "unsafe zip path"):
                source_loader._extract_supported_zip(zip_path, extract_dir)

            self.assertFalse((base_dir / "escape.md").exists())

    def test_safe_zip_extraction_only_extracts_supported_non_ignored_files(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            zip_path = base_dir / "repo.zip"
            extract_dir = base_dir / "extract"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("repo-main/README.md", "# ok\n")
                archive.writestr("repo-main/src/app.py", "print('ok')\n")
                archive.writestr("repo-main/package-lock.json", '{"name":"ignored"}\n')
                archive.writestr("repo-main/node_modules/pkg/index.js", "ignored\n")
                archive.writestr("repo-main/assets/logo.png", "ignored\n")

            root = source_loader._extract_supported_zip(zip_path, extract_dir)

            self.assertEqual(root, extract_dir / "repo-main")
            self.assertTrue((root / "README.md").exists())
            self.assertTrue((root / "src" / "app.py").exists())
            self.assertFalse((root / "package-lock.json").exists())
            self.assertFalse((root / "node_modules" / "pkg" / "index.js").exists())
            self.assertFalse((root / "assets" / "logo.png").exists())


class HuggingFaceSourceLoaderTests(unittest.TestCase):
    def test_is_huggingface_url_model_url(self):
        self.assertTrue(source_loader._is_huggingface_url("https://huggingface.co/meta-llama/Llama-2-7b"))
        self.assertTrue(source_loader._is_huggingface_url("http://huggingface.co/meta-llama/Llama-2-7b"))

    def test_is_huggingface_url_dataset_url(self):
        self.assertTrue(source_loader._is_huggingface_url("https://huggingface.co/datasets/squad/squad"))

    def test_is_huggingface_url_shorthand(self):
        self.assertTrue(source_loader._is_huggingface_url("hf:meta-llama/Llama-2-7b"))

    def test_is_huggingface_url_rejects_github(self):
        self.assertFalse(source_loader._is_huggingface_url("https://github.com/example/project"))

    def test_is_huggingface_url_rejects_local_path(self):
        self.assertFalse(source_loader._is_huggingface_url("/home/user/project"))
        self.assertFalse(source_loader._is_huggingface_url("some-local-dir"))

    def test_huggingface_owner_model_full_url(self):
        owner, model_id = source_loader._huggingface_owner_model("https://huggingface.co/meta-llama/Llama-2-7b")
        self.assertEqual(owner, "meta-llama")
        self.assertEqual(model_id, "Llama-2-7b")

    def test_huggingface_owner_model_dataset_url(self):
        owner, model_id = source_loader._huggingface_owner_model("https://huggingface.co/datasets/squad/squad")
        self.assertEqual(owner, "squad")
        self.assertEqual(model_id, "squad")

    def test_huggingface_owner_model_shorthand(self):
        owner, model_id = source_loader._huggingface_owner_model("hf:meta-llama/Llama-2-7b")
        self.assertEqual(owner, "meta-llama")
        self.assertEqual(model_id, "Llama-2-7b")

    def test_huggingface_owner_model_rejects_short_url(self):
        with self.assertRaisesRegex(ValueError, "must include owner"):
            source_loader._huggingface_owner_model("https://huggingface.co/only-owner")

    def test_huggingface_owner_model_rejects_short_shorthand(self):
        with self.assertRaisesRegex(ValueError, "must include owner"):
            source_loader._huggingface_owner_model("hf:only-owner")

    def test_huggingface_owner_model_rejects_empty_dataset_path(self):
        with self.assertRaisesRegex(ValueError, "must include owner"):
            source_loader._huggingface_owner_model("https://huggingface.co/datasets/only-owner")

    def test_huggingface_target_name_slug(self):
        self.assertEqual(source_loader._huggingface_target_name("meta-llama", "Llama-2-7b"), "meta-llama-Llama-2-7b")

    def test_huggingface_target_name_special_chars(self):
        result = source_loader._huggingface_target_name("my org", "my model v1.0")
        self.assertEqual(result, "my-org-my-model-v1.0")

    def test_huggingface_network_allowed_by_param(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(source_loader._huggingface_network_allowed(True))
            self.assertFalse(source_loader._huggingface_network_allowed(False))

    def test_huggingface_network_allowed_by_env(self):
        with patch.dict(os.environ, {"ALLOW_HF_FETCH": "1"}):
            self.assertTrue(source_loader._huggingface_network_allowed(False))

    def test_huggingface_network_allowed_env_other_value(self):
        with patch.dict(os.environ, {"ALLOW_HF_FETCH": "yes"}):
            self.assertFalse(source_loader._huggingface_network_allowed(False))

    def test_fetch_huggingface_card_model(self):
        with TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir)
            fake_response = MagicMock()
            fake_response.status_code = 200
            fake_response.raise_for_status = MagicMock()
            fake_response.content = b"# Model Card\nA model.\n"

            with patch.object(source_loader.requests, "get", return_value=fake_response) as mock_get:
                result = source_loader._fetch_huggingface_card(
                    "https://huggingface.co/meta-llama/Llama-2-7b",
                    download_dir,
                )

            mock_get.assert_called_once_with(
                "https://huggingface.co/meta-llama/Llama-2-7b/resolve/main/README.md",
                timeout=30,
            )
            self.assertEqual(result.name, "README.md")
            self.assertEqual(result.read_bytes(), b"# Model Card\nA model.\n")
            self.assertEqual(result.parent, download_dir)

    def test_fetch_huggingface_card_dataset(self):
        with TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir)
            fake_response = MagicMock()
            fake_response.status_code = 200
            fake_response.raise_for_status = MagicMock()
            fake_response.content = b"# Dataset Card\nA dataset.\n"

            with patch.object(source_loader.requests, "get", return_value=fake_response) as mock_get:
                result = source_loader._fetch_huggingface_card(
                    "https://huggingface.co/datasets/squad/squad",
                    download_dir,
                )

            mock_get.assert_called_once_with(
                "https://huggingface.co/datasets/squad/squad/resolve/main/README.md",
                timeout=30,
            )
            self.assertEqual(result.name, "README.md")
            self.assertEqual(result.read_bytes(), b"# Dataset Card\nA dataset.\n")

    def test_fetch_huggingface_card_shorthand(self):
        with TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir)
            fake_response = MagicMock()
            fake_response.status_code = 200
            fake_response.raise_for_status = MagicMock()
            fake_response.content = b"# Model\nShorthand test.\n"

            with patch.object(source_loader.requests, "get", return_value=fake_response) as mock_get:
                result = source_loader._fetch_huggingface_card(
                    "hf:meta-llama/Llama-2-7b",
                    download_dir,
                )

            mock_get.assert_called_once_with(
                "https://huggingface.co/meta-llama/Llama-2-7b/resolve/main/README.md",
                timeout=30,
            )
            self.assertEqual(result.name, "README.md")

    def test_fetch_huggingface_card_raises_on_http_error(self):
        import requests as req

        with TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir)
            fake_response = MagicMock()
            fake_response.raise_for_status.side_effect = req.HTTPError("404 Not Found")

            with patch.object(source_loader.requests, "get", return_value=fake_response):
                with self.assertRaises(req.HTTPError):
                    source_loader._fetch_huggingface_card(
                        "https://huggingface.co/nonexistent/model",
                        download_dir,
                    )

    def test_prepare_sources_rejects_hf_url_without_opt_in(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "allow_huggingface_fetch"):
                    source_loader.prepare_sources(
                        ["https://huggingface.co/meta-llama/Llama-2-7b"],
                        raw_dir=raw_dir,
                        allow_huggingface_fetch=False,
                    )

    def test_prepare_sources_rejects_hf_shorthand_without_opt_in(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "allow_huggingface_fetch"):
                    source_loader.prepare_sources(
                        ["hf:meta-llama/Llama-2-7b"],
                        raw_dir=raw_dir,
                    )

    def test_prepare_sources_copies_hf_card_to_raw_dir(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            raw_dir = base_dir / "raw"
            fake_response = MagicMock()
            fake_response.status_code = 200
            fake_response.raise_for_status = MagicMock()
            fake_response.content = b"# Model Card\n\nA great model.\n"

            with (
                patch.object(source_loader.requests, "get", return_value=fake_response),
                patch.object(source_loader, "CURRENT_SOURCE_PATH", base_dir / "current_source.json"),
            ):
                files = source_loader.prepare_sources(
                    ["https://huggingface.co/meta-llama/Llama-2-7b"],
                    raw_dir=raw_dir,
                    allow_huggingface_fetch=True,
                )

            self.assertEqual(len(files), 1)
            relative = files[0].relative_to(raw_dir).as_posix()
            self.assertEqual(relative, "meta-llama-Llama-2-7b/README.md")
            self.assertEqual(files[0].read_text(encoding="utf-8"), "# Model Card\n\nA great model.\n")

    def test_prepare_sources_registers_hf_metadata_without_writing_current_source(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            raw_dir = base_dir / "raw"
            current_source_path = base_dir / "current_source.json"
            fake_response = MagicMock()
            fake_response.status_code = 200
            fake_response.raise_for_status = MagicMock()
            fake_response.content = b"# Model Card\n\nA great model.\n"

            with (
                patch.object(source_loader.requests, "get", return_value=fake_response),
                patch.object(source_loader, "CURRENT_SOURCE_PATH", current_source_path, create=True),
            ):
                files = source_loader.prepare_sources(
                    ["hf:Shizu0n/phi3-mini-sql-generator"],
                    raw_dir=raw_dir,
                    allow_huggingface_fetch=True,
                )

            self.assertFalse(current_source_path.exists())
            metadata = source_loader.PREPARED_SOURCE_METADATA[files[0].resolve().as_posix()]
            self.assertEqual(metadata["source_type"], "huggingface")
            self.assertEqual(metadata["source_input"], "hf:Shizu0n/phi3-mini-sql-generator")
            self.assertEqual(metadata["source_slug"], "shizu0n-phi3-mini-sql-generator")
            self.assertIsNone(metadata.get("indexed_at"))

    def test_prepare_sources_copies_hf_card_via_env(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            raw_dir = base_dir / "raw"
            fake_response = MagicMock()
            fake_response.status_code = 200
            fake_response.raise_for_status = MagicMock()
            fake_response.content = b"# Card\n"

            with patch.dict(os.environ, {"ALLOW_HF_FETCH": "1"}):
                with (
                    patch.object(source_loader.requests, "get", return_value=fake_response),
                    patch.object(source_loader, "CURRENT_SOURCE_PATH", base_dir / "current_source.json"),
                ):
                    files = source_loader.prepare_sources(
                        ["hf:meta-llama/Llama-2-7b"],
                        raw_dir=raw_dir,
                    )

            self.assertEqual(len(files), 1)
            relative = files[0].relative_to(raw_dir).as_posix()
            self.assertEqual(relative, "meta-llama-Llama-2-7b/README.md")

    def test_prepare_sources_hf_mixed_with_local(self):
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            raw_dir = base_dir / "raw"
            local_dir = base_dir / "project"
            local_dir.mkdir()
            (local_dir / "app.py").write_text("print('ok')\n", encoding="utf-8")

            fake_response = MagicMock()
            fake_response.status_code = 200
            fake_response.raise_for_status = MagicMock()
            fake_response.content = b"# HF Card\n"

            with (
                patch.object(source_loader.requests, "get", return_value=fake_response),
                patch.object(source_loader, "CURRENT_SOURCE_PATH", base_dir / "current_source.json"),
            ):
                files = source_loader.prepare_sources(
                    [local_dir, "https://huggingface.co/meta-llama/Llama-2-7b"],
                    raw_dir=raw_dir,
                    allow_huggingface_fetch=True,
                )

            relative_paths = sorted(path.relative_to(raw_dir).as_posix() for path in files)
            self.assertEqual(len(relative_paths), 2)
            self.assertTrue(any("Llama-2-7b/README.md" in p for p in relative_paths))
            self.assertTrue(any("app.py" in p for p in relative_paths))


if __name__ == "__main__":
    unittest.main()
