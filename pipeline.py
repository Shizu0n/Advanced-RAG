"""Local/free RAG pipeline assembly and extractive answer synthesis."""

from __future__ import annotations

import re
import os
import json
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Sequence


SUPPORTED_STRATEGIES = {
    "semantic_only",
    "bm25_only",
    "hybrid_no_rerank",
    "hybrid_rerank",
}
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CONTEXT_PATHS = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "data" / "raw"]
LOCAL_CONTEXT_EXTENSIONS = {
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
LOCAL_CONTEXT_FILENAMES = {"Dockerfile", "Makefile", "Procfile"}
LOCAL_CONTEXT_IGNORED_FILENAMES = {
    "Cargo.lock",
    "Pipfile.lock",
    "composer.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "yarn.lock",
}
LOCAL_CONTEXT_IGNORED_DIR_NAMES = {
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
EMPTY_LOCAL_CONTEXT_ANSWER = "No local context files were found for offline retrieval."
LOW_EVIDENCE_ANSWER = "Nao encontrei evidencia suficiente no contexto recuperado para responder sem inventar."
LOW_EVIDENCE_THRESHOLD = 0.08
KNOWN_TECHNOLOGIES = (
    "React Router",
    "TypeScript",
    "Next.js",
    "NestJS",
    "Express",
    "TypeORM",
    "SQLite",
    "React",
    "Vite",
    "Axios",
    "ESLint",
    "Jest",
    "Prettier",
    "Python",
    "Streamlit",
    "ChromaDB",
    "RAGAS",
    "Gemini",
    "LlamaIndex",
    "BM25",
)
STOPWORDS = {
    "a",
    "about",
    "as",
    "and",
    "are",
    "at",
    "be",
    "by",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "does",
    "dos",
    "e",
    "em",
    "for",
    "from",
    "how",
    "is",
    "it",
    "no",
    "na",
    "nas",
    "nos",
    "o",
    "of",
    "on",
    "or",
    "os",
    "para",
    "por",
    "qual",
    "quais",
    "que",
    "the",
    "this",
    "to",
    "um",
    "uma",
    "use",
    "uses",
    "what",
    "which",
}
INTENT_REWRITES = {
    "stack": "stack tech stack tecnologias ferramentas frameworks dependencies package.json frontend backend react vite nestjs",
    "overview": "overview visao geral resumo objetivo problema solucao free-tier rag workspace local retrieval streamlit evaluation",
    "architecture": "architecture arquitetura modules modulos estrutura fluxo components backend frontend",
    "setup": "setup install instalar executar rodar ambiente env scripts npm",
    "security": "security seguranca auth authentication jwt password senha bcrypt guard token",
    "evaluation": "evaluation avaliacao metricas tests testes qualidade benchmark ragas",
}


@dataclass(frozen=True)
class QueryAnalysis:
    original_query: str
    rewritten_query: str
    intents: list[str]
    terms: set[str]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _normalize_for_match(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _significant_terms(text: str) -> set[str]:
    return {term for term in _tokenize(_normalize_for_match(text)) if term not in STOPWORDS and len(term) > 1}


def analyze_query(query: str) -> QueryAnalysis:
    normalized = _normalize_for_match(query)
    intents: list[str] = []

    intent_patterns = {
        "stack": ("stack", "tech stack", "tecnologia", "tecnologias", "ferramenta", "ferramentas", "framework"),
        "overview": (
            "overview",
            "visao geral",
            "resumo",
            "objetivo",
            "problema",
            "solucao",
            "what is",
            "what this project",
            "project is about",
            "what this repo",
            "repo is about",
            "o que e",
        ),
        "architecture": ("architecture", "arquitetura", "module", "modules", "modulo", "modulos", "estrutura", "fluxo"),
        "setup": ("setup", "install", "instalar", "executar", "rodar", "ambiente", "env", "script", "scripts"),
        "security": ("security", "seguranca", "auth", "authentication", "jwt", "senha", "password", "bcrypt", "token"),
        "evaluation": ("evaluation", "avaliacao", "avaliar", "metric", "metrics", "metrica", "metricas", "test", "tests"),
    }
    for intent, patterns in intent_patterns.items():
        if any(pattern in normalized for pattern in patterns):
            intents.append(intent)

    rewrite_parts = [normalized]
    for intent in intents:
        rewrite_parts.append(INTENT_REWRITES[intent])
    if "front" in normalized:
        rewrite_parts.append("frontend front-end client ui react vite")
    if "back" in normalized:
        rewrite_parts.append("backend server api nestjs typeorm sqlite")

    rewritten_query = " ".join(dict.fromkeys(" ".join(rewrite_parts).split()))
    return QueryAnalysis(
        original_query=query,
        rewritten_query=rewritten_query,
        intents=intents,
        terms=_significant_terms(rewritten_query),
    )


def _node_text(item: Any) -> str:
    node = getattr(item, "node", item)
    if hasattr(node, "get_content"):
        return node.get_content()
    return getattr(node, "text", str(node))


def _source_doc(item: Any) -> str:
    node = getattr(item, "node", item)
    metadata = getattr(node, "metadata", {}) or {}
    return (
        metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("file_path")
        or getattr(node, "node_id", "unknown")
    )


def _score(item: Any) -> float | None:
    value = getattr(item, "score", None)
    return float(value) if value is not None else None


def _snippet(text: str, max_chars: int = 240) -> str:
    compact = " ".join(text.split())
    return compact[: max_chars - 3] + "..." if len(compact) > max_chars else compact


@dataclass
class LocalTextNode:
    text: str
    node_id: str
    metadata: dict[str, Any]

    def get_content(self) -> str:
        return self.text


@dataclass
class LocalNodeWithScore:
    node: LocalTextNode
    score: float


def _lexical_score(query: str, text: str) -> float:
    query_terms = _significant_terms(query)
    text_terms = _significant_terms(text)
    if not query_terms or not text_terms:
        return 0.0

    overlap = len(query_terms & text_terms)
    density = overlap / max(len(text_terms), 1)
    coverage = overlap / max(len(query_terms), 1)
    return coverage + density


def _source_priority(source: str, text: str, analysis: QueryAnalysis) -> float:
    source_key = source.lower().replace("\\", "/")
    text_key = _normalize_for_match(text)
    priority = 0.0

    if "stack" in analysis.intents:
        asks_frontend = "frontend" in analysis.terms or "front" in analysis.terms
        if source_key.endswith("readme.md") and ("stack e ferramentas" in text_key or "stack" in text_key):
            priority += 1.2
            if asks_frontend and "frontend" in text_key:
                priority += 0.8
        if source_key.endswith("frontend/package.json"):
            priority += 1.6 if asks_frontend else 0.7
        if asks_frontend and (
            source_key.endswith("api.ts")
            or ".env" in source_key
            or "setup" in source_key
        ):
            priority -= 0.8

    if "architecture" in analysis.intents and source_key.endswith("readme.md"):
        priority += 0.5
    if "overview" in analysis.intents and source_key.endswith("readme.md"):
        priority += 0.5
        if "free-tier rag workspace" in text_key or "advanced rag" in text_key:
            priority += 1.2
        if "using your own github projects" in text_key or "configuration" in text_key:
            priority -= 0.5
    if "setup" in analysis.intents and ("readme" in source_key or "setup" in source_key or ".env" in source_key):
        priority += 0.5
    if "security" in analysis.intents and any(term in source_key for term in ("auth", "jwt", "guard", "security")):
        priority += 0.6
    if "evaluation" in analysis.intents and any(term in source_key for term in ("eval", "test", "ragas")):
        priority += 0.6

    return priority


def _local_retrieval_score(query: str, text: str, source: str) -> float:
    analysis = analyze_query(query)
    lexical = _lexical_score(analysis.rewritten_query, text)
    return lexical + _source_priority(source, text, analysis)


def _score_local_node(node: Any, analysis: QueryAnalysis) -> float:
    text = _node_text(node)
    return _lexical_score(analysis.rewritten_query, text) + _source_priority(_source_doc(node), text, analysis)


def _chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [text.strip()]:
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _dependency_summary(values: dict[str, Any]) -> str:
    return ", ".join(f"{name} {version}" for name, version in sorted(values.items())) or "none"


def _format_package_manifest(path: Path) -> str | None:
    try:
        data = json.loads(_safe_read_text(path))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    dependencies = data.get("dependencies") or {}
    dev_dependencies = data.get("devDependencies") or {}
    if not isinstance(dependencies, dict):
        dependencies = {}
    if not isinstance(dev_dependencies, dict):
        dev_dependencies = {}

    all_dependency_names = set(dependencies) | set(dev_dependencies)
    framework_hints: list[str] = []
    if any(name.startswith("@nestjs/") for name in all_dependency_names):
        framework_hints.append("NestJS")
    if "express" in all_dependency_names or "@nestjs/platform-express" in all_dependency_names:
        framework_hints.append("Express")
    if "typeorm" in all_dependency_names or "@nestjs/typeorm" in all_dependency_names:
        framework_hints.append("TypeORM")
    if "sqlite3" in all_dependency_names:
        framework_hints.append("SQLite")
    if "vite" in all_dependency_names or "@vitejs/plugin-react" in all_dependency_names:
        framework_hints.append("Vite")
    if "react" in all_dependency_names:
        framework_hints.append("React")
    if "react-router" in all_dependency_names or "react-router-dom" in all_dependency_names:
        framework_hints.append("React Router")
    if "axios" in all_dependency_names:
        framework_hints.append("Axios")
    if "next" in all_dependency_names:
        framework_hints.append("Next.js")
    if "typescript" in all_dependency_names:
        framework_hints.append("TypeScript")
    if "eslint" in all_dependency_names or any(name.startswith("@eslint/") for name in all_dependency_names):
        framework_hints.append("ESLint")
    if "jest" in all_dependency_names or "ts-jest" in all_dependency_names:
        framework_hints.append("Jest")
    if "prettier" in all_dependency_names:
        framework_hints.append("Prettier")

    package_name = str(data.get("name") or path.parent.name)
    package_role = "backend package" if "backend" in package_name.lower() else "frontend package" if "frontend" in package_name.lower() else "package"
    return (
        f"Package manifest for {package_role} {package_name} version {data.get('version', 'unknown')} "
        f"with framework and platform hints: {', '.join(framework_hints) if framework_hints else 'none detected'}. "
        f"Runtime dependencies: {_dependency_summary(dependencies)}. "
        f"Development dependencies: {_dependency_summary(dev_dependencies)}."
    )


def _looks_like_json_text(text: str) -> bool:
    stripped = text.strip()
    return (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))


def _is_mostly_json(text: str, threshold: float = 0.4) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    json_chars = sum(1 for c in stripped if c in "{}[]:;,'\"")
    return (json_chars / len(stripped)) > threshold


def _summarize_json_value(value: Any, max_items: int = 12) -> str:
    if isinstance(value, dict):
        keys = sorted(str(key) for key in value.keys())[:max_items]
        return f"object with keys: {', '.join(keys) if keys else 'none'}"
    if isinstance(value, list):
        sample_types = sorted({type(item).__name__ for item in value[:max_items]})
        return f"array with {len(value)} item(s); sample item types: {', '.join(sample_types) if sample_types else 'none'}"
    if isinstance(value, str):
        compact = " ".join(value.split())
        return compact[:120] + ("..." if len(compact) > 120 else "")
    return str(value)


def _format_json_document(path: Path, text: str | None = None) -> str | None:
    raw_text = _safe_read_text(path) if text is None else text
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    document_name = path.name if path.name else "JSON context"
    if isinstance(data, dict):
        keys = sorted(str(key) for key in data.keys())
        fields: list[str] = []
        for key in keys[:10]:
            value = data.get(key)
            if isinstance(value, (dict, list)):
                fields.append(f"{key}: {_summarize_json_value(value, max_items=8)}")
            elif value is not None:
                fields.append(f"{key}: {_summarize_json_value(value)}")
        return (
            f"JSON document summary for {document_name}. "
            f"Top-level keys: {', '.join(keys[:20]) if keys else 'none'}. "
            f"Selected fields: {'; '.join(fields) if fields else 'none'}."
        )
    if isinstance(data, list):
        return f"JSON document summary for {document_name}. {_summarize_json_value(data)}."
    return f"JSON document summary for {document_name}. Scalar value: {_summarize_json_value(data)}."


def _context_for_synthesis(context: str) -> str:
    if _looks_like_json_text(context):
        return _format_json_document(Path("retrieved.json"), text=context) or "JSON context could not be summarized."
    return context


def _evidence_score(query: str, contexts: Sequence[str]) -> float:
    if not contexts:
        return 0.0
    analysis = analyze_query(query)
    return max(_lexical_score(analysis.rewritten_query, _context_for_synthesis(context)) for context in contexts)


def _confidence(query: str, contexts: Sequence[str], sources: Sequence[dict[str, Any]]) -> float:
    lexical = _evidence_score(query, contexts)
    source_scores = [source.get("score") for source in sources if isinstance(source.get("score"), (int, float))]
    retrieval = max(source_scores) if source_scores else 0.0
    return round(min(1.0, max(lexical / 1.5, retrieval if retrieval <= 1 else retrieval / 5)), 3)


def _has_enough_evidence(query: str, contexts: Sequence[str], sources: Sequence[dict[str, Any]]) -> bool:
    if not contexts:
        return False
    if _evidence_score(query, contexts) >= LOW_EVIDENCE_THRESHOLD:
        return True
    return any(isinstance(source.get("score"), (int, float)) and source["score"] >= LOW_EVIDENCE_THRESHOLD for source in sources)


def _make_citations(results: Sequence[Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for result in results:
        text = _context_for_synthesis(_node_text(result))
        citations.append(
            {
                "source_doc": _source_doc(result),
                "score": _score(result),
                "snippet": _snippet(text),
            }
        )
    return citations


def _extract_technologies(contexts: Sequence[str], query: str = "") -> list[str]:
    combined = "\n".join(_context_for_synthesis(context) for context in contexts)
    normalized = _normalize_for_match(combined)
    query_terms = _significant_terms(query)
    frontend_only = "frontend" in query_terms or "front" in query_terms
    backend_only = "backend" in query_terms or "back" in query_terms
    frontend_tech = {"React Router", "TypeScript", "React", "Vite", "Axios", "ESLint", "Jest", "Prettier"}
    backend_tech = {"NestJS", "Express", "TypeORM", "SQLite"}
    found: list[str] = []
    for tech in KNOWN_TECHNOLOGIES:
        if frontend_only and tech in backend_tech:
            continue
        if backend_only and tech in frontend_tech:
            continue
        if _normalize_for_match(tech) in normalized:
            found.append(tech)
    return found


def _format_stack_answer(query: str, contexts: Sequence[str], citations: Sequence[dict[str, Any]]) -> str:
    technologies = _extract_technologies(contexts, query=query)
    evidence_docs = ", ".join(dict.fromkeys(str(citation["source_doc"]) for citation in citations[:3]))
    if technologies:
        return f"Tecnologias encontradas: {', '.join(technologies)}. Evidencia em: {evidence_docs}."
    return synthesize_extractive_answer("tech stack tecnologias frameworks ferramentas", contexts, max_sentences=3)


def _format_overview_answer(query: str, contexts: Sequence[str], citations: Sequence[dict[str, Any]]) -> str:
    summary = synthesize_extractive_answer(query, contexts, max_sentences=3)
    evidence_docs = ", ".join(dict.fromkeys(str(citation["source_doc"]) for citation in citations[:3]))
    return f"{summary} Evidencia em: {evidence_docs}." if evidence_docs else summary


def synthesize_chat_answer(
    query: str,
    contexts: Sequence[str],
    citations: Sequence[dict[str, Any]],
    intent: str,
    sources: Sequence[dict[str, Any]],
) -> str:
    if not contexts:
        return EMPTY_LOCAL_CONTEXT_ANSWER
    if not _has_enough_evidence(query, contexts, sources):
        return LOW_EVIDENCE_ANSWER
    if intent == "stack":
        return _format_stack_answer(query, contexts, citations)
    if intent == "overview":
        return _format_overview_answer(query, contexts, citations)
    # Tenta síntese gerativa com LLM antes de recorrer ao fallback extraívo
    try:
        from synthesis import synthesize_generative_answer  # noqa: PLC0415

        generative = synthesize_generative_answer(query, contexts, sources)
        if generative is not None:
            return generative
    except Exception:
        pass

    # Fallback extraívo melhorado
    cleaned_contexts = [
        ctx for ctx in contexts if not _is_mostly_json(ctx)
    ]
    return synthesize_extractive_answer(
        query, cleaned_contexts or contexts, max_sentences=5
    )


def _readme_section_snippets(path: Path) -> list[str]:
    text = _safe_read_text(path)
    lines = text.splitlines()
    snippets: list[str] = []
    section_path: list[str] = []
    current_heading = "Overview"
    current_lines: list[str] = []

    def flush() -> None:
        content = "\n".join(line.rstrip() for line in current_lines).strip()
        if not content:
            return
        snippets.append(
            f"README section from {path.name}. "
            f"Section: {' > '.join(section_path) if section_path else current_heading}. "
            f"Content:\n{content}"
        )

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            section_path[:] = section_path[: level - 1]
            section_path.append(heading)
            current_heading = heading
            current_lines = []
            continue
        current_lines.append(line)

    flush()
    return snippets or [text]


def _context_text_for_file(path: Path) -> str:
    if path.name == "package.json":
        manifest_text = _format_package_manifest(path)
        if manifest_text:
            return manifest_text
    if path.suffix.lower() == ".json":
        json_text = _format_json_document(path)
        if json_text:
            return json_text
    return _safe_read_text(path)


def _context_chunks_for_file(path: Path) -> list[str]:
    if path.name.lower() == "readme.md":
        chunks: list[str] = []
        for snippet in _readme_section_snippets(path):
            chunks.extend(_chunk_text(snippet))
        return chunks
    text = _context_text_for_file(path).strip()
    return _chunk_text(text) if text else []


def _is_supported_local_context_file(path: Path) -> bool:
    return (
        path.name not in LOCAL_CONTEXT_IGNORED_FILENAMES
        and path.is_file()
        and (path.suffix.lower() in LOCAL_CONTEXT_EXTENSIONS or path.name in LOCAL_CONTEXT_FILENAMES)
    ) and not any(part in LOCAL_CONTEXT_IGNORED_DIR_NAMES for part in path.parts)


def load_local_context_nodes(paths: Sequence[Path] | None = None) -> list[LocalTextNode]:
    paths = paths if paths is not None else LOCAL_CONTEXT_PATHS
    nodes: list[LocalTextNode] = []
    files: list[Path] = []

    for path in paths:
        if _is_supported_local_context_file(path):
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if _is_supported_local_context_file(candidate)
            )

    for file_path in sorted(set(files)):
        chunks = [chunk for chunk in _context_chunks_for_file(file_path) if re.search(r"\w", chunk)]
        if not chunks:
            continue
        for index, chunk in enumerate(chunks):
            nodes.append(
                LocalTextNode(
                    text=chunk,
                    node_id=f"{file_path.as_posix()}#{index}",
                    metadata={
                        "file_name": file_path.relative_to(PROJECT_ROOT).as_posix()
                        if file_path.is_relative_to(PROJECT_ROOT)
                        else file_path.as_posix(),
                        "source": "local_context",
                    },
                )
            )
    return nodes


def synthesize_extractive_answer(query: str, contexts: Sequence[str], max_sentences: int = 3) -> str:
    """Pick the context sentences with the strongest lexical overlap with the query."""

    query_terms = _tokenize(query)
    ranked: list[tuple[float, int, str]] = []

    for context_index, context in enumerate(contexts):
        context = _context_for_synthesis(context)
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", context) if part.strip()]
        for sentence_index, sentence in enumerate(sentences or [context.strip()]):
            terms = _tokenize(sentence)
            overlap = len(query_terms & terms)
            density = overlap / max(len(terms), 1)
            ranked.append((overlap + density, context_index * 1000 + sentence_index, sentence))

    if not ranked:
        return "No relevant context was retrieved."

    best = sorted(ranked, key=lambda item: (-item[0], item[1]))[:max_sentences]
    answer = " ".join(sentence for _, _, sentence in sorted(best, key=lambda item: item[1]))
    return answer or contexts[0][:500]


class LocalLexicalCrossEncoder:
    """No-cost reranker compatible with sentence-transformers CrossEncoder.predict."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, text in pairs:
            query_terms = _tokenize(query)
            text_terms = _tokenize(text)
            overlap = len(query_terms & text_terms)
            scores.append(overlap / max(len(query_terms), 1))
        return scores


class SimpleLocalRetriever:
    """Fallback retriever for tests or offline use without a vector index."""

    def __init__(self, nodes: Sequence[Any], top_k: int = 5) -> None:
        self.nodes = list(nodes)
        self.top_k = top_k

    def ablation_retrieve(self, query: str, strategy: str) -> tuple[list[Any], dict[str, Any]]:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")

        analysis = analyze_query(query)
        scored = [
            LocalNodeWithScore(node=getattr(node, "node", node), score=_score_local_node(node, analysis))
            for node in self.nodes
        ]
        ranked = sorted(scored, key=lambda node: node.score, reverse=True)
        results = [node for node in ranked[: self.top_k] if node.score > 0]
        if not results:
            results = ranked[: self.top_k]
        score_rows = [
            {"source": _source_doc(result), "score": result.score}
            for result in results
        ]
        return results, {
            "strategy": strategy,
            "top_k": self.top_k,
            "fallback": "local_lexical",
            "intents": analysis.intents,
            "rewritten_query": analysis.rewritten_query,
            "query_terms": sorted(analysis.terms),
            "used_vector": False,
            "used_bm25": False,
            "used_rerank": False,
            "reranker": None,
            "lexical_scores": score_rows,
            "bm25_scores": [],
            "vector_scores": [],
            "rrf_scores": [],
            "reranker_scores": [],
        }


@dataclass
class LocalRAGPipeline:
    index: Any | None = None
    nodes: Sequence[Any] | None = None
    retriever: Any | None = None
    top_k: int = 5
    allow_index_build: bool = False

    def __post_init__(self) -> None:
        self.nodes = list(self.nodes or [])
        if self.retriever is None and (self.index is not None or self.nodes):
            self.retriever = self._build_retriever()

    def _ensure_ready(self) -> None:
        if self.retriever is not None:
            return

        if not self.allow_index_build:
            self.nodes = load_local_context_nodes()
            self.retriever = SimpleLocalRetriever(self.nodes, top_k=self.top_k)
            return

        if os.getenv("ALLOW_INDEX_BUILD") != "1":
            raise RuntimeError(
                "Index build requested but ALLOW_INDEX_BUILD is not set to 1. "
                "This prevents accidental scraping or model downloads."
            )

        from ingestion import build_index

        self.index, self.nodes = build_index()
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        if self.index is None:
            return SimpleLocalRetriever(self.nodes or [], top_k=self.top_k)

        from retrieval import HybridRetriever

        return HybridRetriever(
            self.index,
            self.nodes or [],
            top_k=self.top_k,
            cross_encoder=LocalLexicalCrossEncoder(),
            cross_encoder_model="local_lexical",
        )

    def _score_rows(self, results: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            {"source": _source_doc(result), "score": _score(result)}
            for result in results
        ]

    def _source_rows(self, results: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            {
                "source_doc": _source_doc(result),
                "score": _score(result),
                "text": _node_text(result),
            }
            for result in results
        ]

    def _normalize_trace(self, metadata: dict[str, Any], results: Sequence[Any]) -> dict[str, Any]:
        source_scores = self._score_rows(results)
        strategy = metadata.get("strategy")
        derived = {
            "bm25_scores": source_scores if strategy == "bm25_only" else [],
            "vector_scores": source_scores if strategy == "semantic_only" else [],
            "rrf_scores": source_scores if strategy == "hybrid_no_rerank" else [],
            "reranker_scores": source_scores if strategy == "hybrid_rerank" else [],
        }
        return {
            "bm25_scores": metadata.get("bm25_scores", derived["bm25_scores"]),
            "vector_scores": metadata.get("vector_scores", derived["vector_scores"]),
            "rrf_scores": metadata.get("rrf_scores", derived["rrf_scores"]),
            "reranker_scores": metadata.get("reranker_scores", derived["reranker_scores"]),
            **metadata,
        }

    def answer_query(self, query: str, strategy: str = "hybrid_rerank") -> dict[str, Any]:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"strategy must be one of: {', '.join(sorted(SUPPORTED_STRATEGIES))}")

        self._ensure_ready()
        results, metadata = self.retriever.ablation_retrieve(query, strategy)
        contexts = [_node_text(result) for result in results]
        answer = synthesize_extractive_answer(query, contexts) if contexts else EMPTY_LOCAL_CONTEXT_ANSWER
        sources = self._source_rows(results)

        return {
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "strategy": strategy,
            "trace": self._normalize_trace(metadata, results),
        }

    def chat_query(
        self,
        message: str,
        history: Sequence[dict[str, Any]] | None = None,
        strategy: str = "hybrid_rerank",
    ) -> dict[str, Any]:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"strategy must be one of: {', '.join(sorted(SUPPORTED_STRATEGIES))}")

        self._ensure_ready()
        history = list(history or [])
        history_tail = " ".join(
            str(item.get("content") or item.get("message") or "")
            for item in history[-3:]
            if isinstance(item, dict)
        )
        retrieval_query = f"{history_tail} {message}".strip() if history_tail else message
        analysis = analyze_query(retrieval_query)
        intent = analysis.intents[0] if analysis.intents else "general"

        results, metadata = self.retriever.ablation_retrieve(retrieval_query, strategy)
        contexts = [_node_text(result) for result in results]
        sources = self._source_rows(results)
        citations = _make_citations(results)
        answer = synthesize_chat_answer(message, contexts, citations, intent, sources)
        trace = self._normalize_trace(metadata, results)
        trace.update(
            {
                "message": message,
                "history_items": len(history),
                "retrieval_query": retrieval_query,
                "intent": intent,
                "intents": analysis.intents,
                "evidence_score": round(_evidence_score(message, contexts), 3),
            }
        )

        return {
            "answer": answer,
            "citations": citations,
            "contexts": contexts,
            "sources": sources,
            "trace": trace,
            "intent": intent,
            "confidence": _confidence(message, contexts, sources),
        }


def answer_query(query: str, strategy: str = "hybrid_rerank", allow_index_build: bool = False) -> dict[str, Any]:
    return LocalRAGPipeline(allow_index_build=allow_index_build).answer_query(query, strategy=strategy)


def chat_query(
    message: str,
    history: Sequence[dict[str, Any]] | None = None,
    strategy: str = "hybrid_rerank",
    allow_index_build: bool = False,
) -> dict[str, Any]:
    return LocalRAGPipeline(allow_index_build=allow_index_build).chat_query(
        message,
        history=history,
        strategy=strategy,
    )
