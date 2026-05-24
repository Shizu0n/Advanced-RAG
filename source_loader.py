"""Prepare local, GitHub repository, and HuggingFace sources for ingestion."""

from __future__ import annotations

import json
import logging
import os
import re
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import requests

logger = logging.getLogger(__name__)


SOURCE_EXTENSIONS = {
    ".c",
    ".cfg",
    ".cs",
    ".cpp",
    ".cxx",
    ".go",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".md",
    ".mjs",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".rst",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
SOURCE_FILENAMES = {"Dockerfile", "Makefile", "Procfile"}
IGNORED_FILE_NAMES = {
    "Cargo.lock",
    "Pipfile.lock",
    "composer.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "yarn.lock",
}
IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".idea",
    ".svn",
    ".venv",
    ".vscode",
    "build",
    "chroma_db",
    "coverage",
    "dist",
    "env",
    "node_modules",
    "target",
    "venv",
    "__pycache__",
}
MAX_GITHUB_ZIP_BYTES = 50 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent
CURRENT_SOURCE_PATH = PROJECT_ROOT / "data" / "current_source.json"
PREPARED_SOURCE_PATH = PROJECT_ROOT / "data" / "prepared_source.json"
PREPARED_SOURCE_METADATA: dict[str, dict[str, str | int | None]] = {}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "source"


def _path_hash(path: Path) -> str:
    resolved = str(path.resolve(strict=False)).lower()
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]


def _local_target_name(path: Path) -> str:
    return f"{_slug(path.name)}-{_path_hash(path)}"


def _github_target_name(owner: str, repo: str) -> str:
    return _slug(f"{owner}-{repo}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_github_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() == "github.com"


def _github_owner_repo(source: str) -> tuple[str, str]:
    parsed = urlparse(source)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"GitHub source must include owner and repository: {source}")

    owner = parts[0]
    repo = parts[1].removesuffix(".git")
    return owner, repo


def _network_allowed(allow_github_fetch: bool) -> bool:
    return allow_github_fetch or os.getenv("ALLOW_GITHUB_FETCH") == "1"


def _is_supported_file(path: Path) -> bool:
    return (
        path.name not in IGNORED_FILE_NAMES
        and not path.is_symlink()
        and path.is_file()
        and (path.suffix.lower() in SOURCE_EXTENSIONS or path.name in SOURCE_FILENAMES)
    )


def _iter_source_files(source_dir: Path, raw_dir: Path | None = None) -> list[Path]:
    files: list[Path] = []
    resolved_raw_dir = raw_dir.resolve(strict=False) if raw_dir is not None else None
    for root, dir_names, file_names in os.walk(source_dir):
        root_path = Path(root)
        if resolved_raw_dir is not None and _is_relative_to(
            root_path.resolve(strict=False), resolved_raw_dir
        ):
            dir_names[:] = []
            continue

        dir_names[:] = [name for name in dir_names if name not in IGNORED_DIR_NAMES]
        if resolved_raw_dir is not None:
            dir_names[:] = [
                name
                for name in dir_names
                if not _is_relative_to((root_path / name).resolve(strict=False), resolved_raw_dir)
            ]

        for file_name in file_names:
            candidate = root_path / file_name
            if _is_supported_file(candidate):
                files.append(candidate)
    return sorted(files)


def _copy_without_conflict(source_file: Path, target_file: Path) -> None:
    if target_file.exists():
        if target_file.read_bytes() != source_file.read_bytes():
            raise FileExistsError(f"Refusing to overwrite conflicting prepared source: {target_file}")
        return

    shutil.copy2(source_file, target_file)


def _copy_source_tree(source_dir: Path, raw_dir: Path, target_name: str) -> list[Path]:
    target_root = raw_dir / _slug(target_name)
    copied_files: list[Path] = []

    for source_file in _iter_source_files(source_dir, raw_dir=raw_dir):
        relative_path = source_file.relative_to(source_dir)
        target_file = target_root / relative_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        _copy_without_conflict(source_file, target_file)
        copied_files.append(target_file)

    return sorted(copied_files)


def _download_limited(url: str, target: Path, max_bytes: int) -> None:
    request = Request(url, headers={"User-Agent": "advanced-rag-source-loader/1.0"})
    total_bytes = 0
    with urlopen(request, timeout=30) as response:
        with target.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise RuntimeError(f"GitHub zipball exceeds maximum size of {max_bytes} bytes.")
                output.write(chunk)


def _archive_member_supported(member_name: str) -> bool:
    member_path = Path(member_name)
    return member_path.name not in IGNORED_FILE_NAMES and not any(part in IGNORED_DIR_NAMES for part in member_path.parts) and (
        member_path.suffix.lower() in SOURCE_EXTENSIONS or member_path.name in SOURCE_FILENAMES
    )


def _extract_supported_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    resolved_extract_dir = extract_dir.resolve(strict=False)
    extracted_files: list[Path] = []

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            target_path = extract_dir / member.filename
            resolved_target_path = target_path.resolve(strict=False)
            if not _is_relative_to(resolved_target_path, resolved_extract_dir):
                raise RuntimeError(f"Refusing unsafe zip path: {member.filename}")

            if not _archive_member_supported(member.filename):
                continue

            resolved_target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, resolved_target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_files.append(target_path)

    if not extracted_files:
        return extract_dir

    roots = {path.relative_to(extract_dir).parts[0] for path in extracted_files}
    return extract_dir / next(iter(roots)) if len(roots) == 1 else extract_dir


def _fetch_github_repository(
    source: str,
    download_dir: Path,
    max_zip_bytes: int = MAX_GITHUB_ZIP_BYTES,
) -> Path:
    owner, repo = _github_owner_repo(source)
    url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    logger.info("Fetching GitHub repo: %s/%s", owner, repo)
    zip_path = download_dir / f"{_slug(owner)}-{_slug(repo)}.zip"

    extract_dir = download_dir / f"{_slug(owner)}-{_slug(repo)}"
    _download_limited(url, zip_path, max_zip_bytes)
    return _extract_supported_zip(zip_path, extract_dir)


def _is_huggingface_url(source: str) -> bool:
    """Return True if *source* is a HuggingFace URL or ``hf:`` shorthand."""
    if source.startswith("hf:"):
        return True
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() == "huggingface.co"


def _huggingface_owner_model(source: str) -> tuple[str, str]:
    """Parse ``(owner, model_id)`` from a HuggingFace URL or ``hf:`` shorthand."""
    if source.startswith("hf:"):
        path = source.removeprefix("hf:")
        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"HuggingFace shorthand must include owner and model id: {source}")
        return parts[0], parts[1]

    parsed = urlparse(source)
    parts = [p for p in parsed.path.split("/") if p]
    if "datasets" in parts:
        idx = parts.index("datasets")
        if len(parts) < idx + 3:
            raise ValueError(f"HuggingFace dataset URL must include owner and dataset id: {source}")
        return parts[idx + 1], parts[idx + 2]
    if len(parts) < 2:
        raise ValueError(f"HuggingFace source must include owner and model id: {source}")
    return parts[0], parts[1]


def _huggingface_target_name(owner: str, model_id: str) -> str:
    """Return a slug for the raw directory name."""
    return _slug(f"{owner}-{model_id}")


def _huggingface_network_allowed(allow_huggingface_fetch: bool) -> bool:
    """Return True if HF fetching is permitted via param or env var."""
    return allow_huggingface_fetch or os.getenv("ALLOW_HF_FETCH") == "1"


def _prepared_metadata(
    *,
    source_input: str,
    source_type: str,
    source_slug: str,
    file_count: int,
) -> dict[str, str | int | None]:
    return {
        "source_input": source_input,
        "source_type": source_type,
        "source_slug": source_slug.lower(),
        "indexed_at": None,
        "file_count": file_count,
        "chunk_count": 0,
    }


def prepared_source_path_for_raw_dir(raw_dir: Path) -> Path:
    return raw_dir.parent / "prepared_source.json"


def _persist_prepared_source(metadata: dict[str, str | int | None], path: Path | None = None) -> None:
    target = path or PREPARED_SOURCE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prepared_source(path: Path | None = None) -> dict | None:
    target = path or PREPARED_SOURCE_PATH
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def clear_prepared_source(path: Path | None = None) -> None:
    target = path or PREPARED_SOURCE_PATH
    if target.exists():
        target.unlink()


def _register_prepared_files(
    files: list[Path],
    *,
    source_input: str,
    source_type: str,
    source_slug: str,
    prepared_source_path: Path | None = None,
) -> dict[str, str | int | None]:
    metadata = _prepared_metadata(
        source_input=source_input,
        source_type=source_type,
        source_slug=source_slug,
        file_count=len(files),
    )
    for file_path in files:
        PREPARED_SOURCE_METADATA[file_path.resolve().as_posix()] = metadata
    _persist_prepared_source(metadata, prepared_source_path)
    return metadata


def _reset_raw_dir(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for path in raw_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _fetch_huggingface_card(
    source: str,
    download_dir: Path,
) -> Path:
    """Fetch the README.md model/dataset card from HuggingFace.

    Uses the ``/resolve/main/README.md`` endpoint so the file is returned as
    raw markdown rather than an HTML page.
    """
    owner, model_id = _huggingface_owner_model(source)
    parsed = urlparse(source)
    is_dataset = not source.startswith("hf:") and "datasets" in parsed.path
    base = "https://huggingface.co"
    if is_dataset:
        url = f"{base}/datasets/{owner}/{model_id}/resolve/main/README.md"
    else:
        url = f"{base}/{owner}/{model_id}/resolve/main/README.md"
    logger.info("Fetching HuggingFace card: %s", url)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    readme_path = download_dir / "README.md"
    readme_path.write_bytes(response.content)
    return readme_path


def prepare_sources(
    sources: list[str | Path] | tuple[str | Path, ...],
    raw_dir: Path | str = PROJECT_ROOT / "data" / "raw",
    allow_github_fetch: bool = False,
    allow_huggingface_fetch: bool = False,
    clear_existing: bool = False,
) -> list[Path]:
    """Copy supported local/GitHub/HuggingFace source files into raw_dir and return copied paths."""

    logger.info("prepare_sources: %d sources, raw_dir=%s", len(sources), raw_dir)
    raw_dir = Path(raw_dir)
    prepared_source_path = prepared_source_path_for_raw_dir(raw_dir)
    if clear_existing:
        _reset_raw_dir(raw_dir)
        PREPARED_SOURCE_METADATA.clear()
        clear_prepared_source(prepared_source_path)
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
    prepared_files: list[Path] = []

    for source in sources:
        source_text = str(source)
        if _is_huggingface_url(source_text):
            owner, model_id = _huggingface_owner_model(source_text)
            if not _huggingface_network_allowed(allow_huggingface_fetch):
                raise RuntimeError(
                    "HuggingFace source fetching requires allow_huggingface_fetch=True "
                    "or ALLOW_HF_FETCH=1."
                )
            with tempfile.TemporaryDirectory() as tmpdir:
                fetched = _fetch_huggingface_card(source_text, Path(tmpdir))
                target_root = raw_dir / _huggingface_target_name(owner, model_id)
                target_root.mkdir(parents=True, exist_ok=True)
                target_file = target_root / fetched.name
                _copy_without_conflict(fetched, target_file)
                source_files = [target_file]
                _register_prepared_files(
                    source_files,
                    source_input=source_text,
                    source_type="huggingface",
                    source_slug=target_root.name,
                    prepared_source_path=prepared_source_path,
                )
                prepared_files.extend(source_files)
            continue
        if _is_github_url(source_text):
            owner, repo = _github_owner_repo(source_text)
            if not _network_allowed(allow_github_fetch):
                raise RuntimeError(
                    "GitHub source fetching requires allow_github_fetch=True "
                    "or ALLOW_GITHUB_FETCH=1."
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                fetched_dir = _fetch_github_repository(source_text, Path(tmpdir))
                source_files = _copy_source_tree(fetched_dir, raw_dir, _github_target_name(owner, repo))
                _register_prepared_files(
                    source_files,
                    source_input=source_text,
                    source_type="github",
                    source_slug=_github_target_name(owner, repo),
                    prepared_source_path=prepared_source_path,
                )
                prepared_files.extend(source_files)
            continue

        source_path = Path(source).expanduser()
        if source_path.is_dir():
            source_files = _copy_source_tree(source_path, raw_dir, _local_target_name(source_path))
            _register_prepared_files(
                source_files,
                source_input=source_text,
                source_type="local",
                source_slug=_local_target_name(source_path),
                prepared_source_path=prepared_source_path,
            )
            prepared_files.extend(source_files)
            continue
        if source_path.is_file() and _is_supported_file(source_path):
            target_root = raw_dir / _local_target_name(source_path)
            target_file = target_root / source_path.name
            target_file.parent.mkdir(parents=True, exist_ok=True)
            _copy_without_conflict(source_path, target_file)
            source_files = [target_file]
            _register_prepared_files(
                source_files,
                source_input=source_text,
                source_type="local",
                source_slug=_local_target_name(source_path),
                prepared_source_path=prepared_source_path,
            )
            prepared_files.extend(source_files)
            continue

        raise FileNotFoundError(f"Unsupported or missing source: {source}")

    return sorted(prepared_files)
