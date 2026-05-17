"""Loads and indexes documents into ChromaDB."""

from __future__ import annotations

import re
import os
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import chromadb
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from source_loader import prepare_sources


RAW_DIR = Path("data/raw")
CHROMA_DIR = Path("chroma_db")
PYTHON_TUTORIAL_URL = "https://docs.python.org/3/tutorial/"
PAGE_LIMIT = 50
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
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


def _safe_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", path).strip("_")
    return f"{filename or 'python_tutorial_index'}.txt"


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _is_supported_source_file(path: Path) -> bool:
    return path.is_file() and (
        path.suffix.lower() in SOURCE_EXTENSIONS or path.name in SOURCE_FILENAMES
    ) and not any(part in IGNORED_DIR_NAMES for part in path.parts)


def _extract_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()

    title = soup.find("h1")
    body = soup.find("div", class_="body") or soup.find("main") or soup.body or soup
    lines = [line.strip() for line in body.get_text("\n").splitlines()]
    text = "\n".join(line for line in lines if line)

    if title and title.get_text(strip=True) not in text[:200]:
        text = f"{title.get_text(strip=True)}\n\n{text}"
    return text


def _discover_tutorial_pages(session: requests.Session, limit: int = PAGE_LIMIT) -> list[str]:
    pending = [PYTHON_TUTORIAL_URL]
    visited: set[str] = set()
    pages: list[str] = []

    while pending and len(pages) < limit:
        url = pending.pop(0)
        if url in visited:
            continue
        visited.add(url)

        response = session.get(url, timeout=20)
        response.raise_for_status()
        pages.append(url)

        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.select("a[href]"):
            next_url = urljoin(url, anchor["href"]).split("#", 1)[0]
            parsed = urlparse(next_url)
            if parsed.netloc != "docs.python.org":
                continue
            if not parsed.path.startswith("/3/tutorial/"):
                continue
            if parsed.path.endswith("/") or parsed.path.endswith(".html"):
                if next_url not in visited and next_url not in pending:
                    pending.append(next_url)

    return pages


def download_python_tutorial_pages(raw_dir: Path = RAW_DIR, limit: int = PAGE_LIMIT) -> list[Path]:
    """Download Python tutorial pages as plain text files."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "advanced-rag-ingestion/1.0"})

    saved_files: list[Path] = []
    for url in _discover_tutorial_pages(session, limit=limit):
        response = session.get(url, timeout=20)
        response.raise_for_status()
        text = _extract_page_text(response.text)
        target = raw_dir / _safe_filename(url)
        target.write_text(f"Source: {url}\n\n{text}\n", encoding="utf-8")
        saved_files.append(target)

    return saved_files


def _source_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    if not raw_dir.exists():
        return []

    return sorted(path for path in raw_dir.rglob("*") if _is_supported_source_file(path))


def load_or_download_sources(raw_dir: Path = RAW_DIR, limit: int = PAGE_LIMIT) -> list[Path]:
    """Use local text/markdown sources when present; otherwise scrape Python docs."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = _source_files(raw_dir)
    if existing:
        return existing
    if os.getenv("ALLOW_DOCS_DOWNLOAD") != "1":
        raise RuntimeError(
            "No source files found in data/raw. Add repository files with prepare_sources(), "
            "or set ALLOW_DOCS_DOWNLOAD=1 to download Python tutorial docs explicitly."
        )
    return download_python_tutorial_pages(raw_dir=raw_dir, limit=limit)


def build_index(source_files: Iterable[Path] | None = None) -> tuple[VectorStoreIndex, list]:
    """Chunk source files, embed them locally, and persist vectors in ChromaDB."""

    files = list(source_files) if source_files is not None else load_or_download_sources()
    if not files:
        raise RuntimeError("No source documents found in data/raw and download produced no files.")

    documents = []
    for path in files:
        text = _safe_read_text(path).strip()
        if not re.search(r"\w", text):
            continue
        documents.append(
            Document(
                text=text,
                metadata={
                    "file_name": path.as_posix(),
                    "source": "repo_source",
                },
            )
        )

    if not documents:
        raise RuntimeError("No readable text documents were found in the provided source files.")

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("advanced_rag")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    start = time.perf_counter()
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    embedding_time = time.perf_counter() - start

    print(f"total documents: {len(documents)}")
    print(f"total chunks: {len(nodes)}")
    print(f"embedding time: {embedding_time:.2f}s")

    return index, nodes


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
