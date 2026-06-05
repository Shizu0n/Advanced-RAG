"""Local/free RAG pipeline assembly and extractive answer synthesis."""

from __future__ import annotations

import logging
import re
import os
import json
import unicodedata
import hashlib
import time
import concurrent.futures

logger = logging.getLogger(__name__)
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
CODE_EVIDENCE_EXTENSIONS = {
    ".c",
    ".cs",
    ".cpp",
    ".cxx",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".mjs",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".ts",
    ".tsx",
}
CODE_EVIDENCE_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "Procfile",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
}
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
    "React Router DOM",
    "React Router",
    "TypeScript",
    "Next.js",
    "NestJS",
    "Express",
    "TypeORM",
    "SQLite3",
    "SQLite",
    "JWT",
    "class-validator",
    "class-transformer",
    "bcrypt",
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
LOW_VALUE_CODE_QUERY_TERMS = {
    "async",
    "const",
    "function",
    "get",
    "return",
    "set",
    "static",
    "string",
}
INTENT_REWRITES = {
    "stack": "stack tech stack tecnologias ferramentas frameworks dependencies package.json frontend backend react vite nestjs",
    "overview": "overview visao geral resumo objetivo problema solucao free-tier rag workspace local retrieval streamlit evaluation",
    "architecture": "architecture arquitetura mvc layer layers camada camadas modules modulos estrutura fluxo components backend frontend service services controller controllers repository repositories model models view views",
    "setup": "setup install instalar executar rodar ambiente env scripts npm deploy deployment docker compose build",
    "security": "security seguranca auth authentication jwt password senha bcrypt guard token",
    "evaluation": "evaluation avaliacao metricas tests testes qualidade benchmark ragas",
    "fine_tune": "fine tune training dataset hyperparameters lora qlora phi training_details",
}
ARCHITECTURE_LAYER_ALIASES = {
    "adapter": {"adapter", "adapters", "gateway", "gateways", "integration", "integrations"},
    "adapters": {"adapter", "adapters", "gateway", "gateways", "integration", "integrations"},
    "aplicacao": {"application", "usecase", "usecases", "interactor", "service"},
    "application": {"application", "usecase", "usecases", "interactor", "service"},
    "app": {"application", "usecase", "usecases", "interactor", "service"},
    "core": {"core", "domain", "shared"},
    "domain": {"domain", "entity", "entities", "model", "models", "aggregate", "aggregates"},
    "dominio": {"domain", "entity", "entities", "model", "models", "aggregate", "aggregates"},
    "infra": {"infrastructure", "repository", "repositories", "persistence", "database", "adapter", "adapters"},
    "infrastructure": {"infrastructure", "repository", "repositories", "persistence", "database", "adapter", "adapters"},
    "infraestrutura": {"infrastructure", "repository", "repositories", "persistence", "database", "adapter", "adapters"},
    "interface": {"interface", "interfaces", "presentation", "controller", "controllers", "route", "routes", "api"},
    "interfaces": {"interface", "interfaces", "presentation", "controller", "controllers", "route", "routes", "api"},
    "mvc": {"model", "models", "view", "views", "controller", "controllers", "route", "routes"},
    "presentation": {"presentation", "controller", "controllers", "route", "routes", "api", "view", "views"},
    "usecase": {"usecase", "usecases", "application", "interactor", "service"},
    "usecases": {"usecase", "usecases", "application", "interactor", "service"},
}
ARCHITECTURE_LAYER_GROUPS = {
    "domain": {"domain", "entity", "entities", "model", "models", "aggregate", "aggregates", "core"},
    "application": {"application", "app", "usecase", "usecases", "interactor", "service", "services", "command", "commands", "query", "queries", "handler", "handlers"},
    "infrastructure": {"infrastructure", "infra", "repository", "repositories", "persistence", "database", "adapter", "adapters", "gateway", "gateways", "client", "clients"},
    "presentation": {"presentation", "interface", "interfaces", "controller", "controllers", "route", "routes", "api", "view", "views", "page", "pages", "component", "components"},
}
SYNTHESIS_SUCCESS = "success"
SYNTHESIS_NO_CONTEXTS = "no_contexts"
SYNTHESIS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SYNTHESIS_CLOUD_CHAT_DISABLED = "cloud_chat_disabled"
SYNTHESIS_NO_PROVIDER_CONFIGURED = "no_provider_configured"
SYNTHESIS_BUDGET_EXCEEDED = "budget_exceeded"
SYNTHESIS_PROVIDER_TIMEOUT = "provider_timeout"
SYNTHESIS_PROVIDER_EXHAUSTED = "provider_exhausted"


@dataclass(frozen=True)
class QueryAnalysis:
    original_query: str
    rewritten_query: str
    intents: list[str]
    terms: set[str]


@dataclass(frozen=True)
class SynthesisError:
    code: str
    stage: str
    retryable: bool
    provider: str | None = None

    def to_trace(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "retryable": self.retryable,
            "provider": self.provider,
        }


@dataclass(frozen=True)
class SynthesisResult:
    answer: str
    mode: str
    error: SynthesisError | None = None
    provider_chain: list[str] | None = None
    provider_timeout_seconds: float | None = None
    total_timeout_seconds: float | None = None
    provider_attempts: list[dict[str, Any]] | None = None

    @property
    def code(self) -> str:
        return self.error.code if self.error else SYNTHESIS_SUCCESS

    def to_trace(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "code": self.code,
            "synthesis_error": self.error.to_trace() if self.error else None,
            "provider_chain": self.provider_chain or [],
            "provider_timeout_seconds": self.provider_timeout_seconds,
            "total_timeout_seconds": self.total_timeout_seconds,
            "provider_attempts": self.provider_attempts or [],
        }


def _env_enabled_by_default(name: str) -> bool:
    return os.getenv(name, "1") != "0"


@dataclass(frozen=True)
class ChatProviderPolicy:
    enabled: bool
    providers: tuple[Any, ...]
    max_calls: int = 1
    provider_timeout_seconds: float = 30.0
    total_timeout_seconds: float = 60.0
    cache_enabled: bool = False

    @classmethod
    def from_env(cls) -> "ChatProviderPolicy":
        if not _env_enabled_by_default("ALLOW_CLOUD_CHAT"):
            return cls(enabled=False, providers=())
        import cloud_ragas

        providers = tuple(cloud_ragas.providers_from_env())
        return cls(
            enabled=True,
            providers=providers,
            max_calls=int(os.getenv("MAX_CLOUD_CHAT_CALLS", str(max(len(providers), 1)))),
            provider_timeout_seconds=float(os.getenv("CLOUD_CHAT_PROVIDER_TIMEOUT_SECONDS", "30")),
            total_timeout_seconds=float(os.getenv("CLOUD_CHAT_TOTAL_TIMEOUT_SECONDS", "60")),
            cache_enabled=os.getenv("CLOUD_CHAT_CACHE", "0") == "1",
        )

    def provider_chain(self) -> list[str]:
        return [provider.name for provider in self.providers]


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9_]+", text):
        parts = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_", " ", raw).split()
        tokens.update(part.lower() for part in parts if part)
    return tokens


def _normalize_for_match(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _significant_terms(text: str) -> set[str]:
    terms = _tokenize(text) | _tokenize(_normalize_for_match(text))
    return {term for term in terms if term not in STOPWORDS and len(term) > 1}


def _expand_architecture_layer_terms(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for term in list(terms):
        expanded.update(ARCHITECTURE_LAYER_ALIASES.get(term, set()))
    return expanded


def _source_path_terms(source: str) -> set[str]:
    source_key = source.lower().replace("\\", "/")
    return _significant_terms(source_key.replace("/", " ").replace(".", " ").replace("-", " "))


def _architecture_layer_bucket(source: str) -> str | None:
    source_terms = _source_path_terms(source)
    for bucket, terms in ARCHITECTURE_LAYER_GROUPS.items():
        if source_terms & terms:
            return bucket
    return None


def _requested_architecture_layer_count(analysis: QueryAnalysis) -> int:
    return sum(1 for terms in ARCHITECTURE_LAYER_GROUPS.values() if analysis.terms & terms)


CODE_REQUEST_PATTERNS = (
    r"\bcode\b",
    r"\bcodigo\b",
    r"\bcódigo\b",
    r"\bsnippet\b",
    r"\bsnippets\b",
    r"\bexample\b",
    r"\bexemplo\b",
    r"\bfunction\b",
    r"\bfuncao\b",
    r"\bfunção\b",
    r"\bclass\b",
    r"\bscript\b",
    r"\bcommand\b",
    r"\bcommands\b",
    r"\bcomando\b",
    r"\bcomandos\b",
    r"\brun\b.*\bscript\b",
    r"\bscripts?\b",
    r"\bhow to run\b",
    r"\bhow to build\b",
    r"\bnpm run\b",
    r"\bstep.*run\b",
    r"\bshow code\b",
    r"\bcode example\b",
    r"\bexample code\b",
    r"\bpython code\b",
    r"\bjavascript code\b",
    r"\btypescript code\b",
    r"\bbash code\b",
    r"\bshell code\b",
    r"\bsource code\b",
    r"\bcode sample\b",
    r"\bsample code\b",
)

SOURCE_CODE_REQUEST_PATTERNS = (
    r"\bsource code\b",
    r"\bshow code\b",
    r"\bcode example\b",
    r"\bexample code\b",
    r"\bpython code\b",
    r"\bjavascript code\b",
    r"\btypescript code\b",
    r"\bbash code\b",
    r"\bshell code\b",
    r"\bcode sample\b",
    r"\bsample code\b",
    r"\bfrom the code\b",
    r"\bfrom code\b",
    r"\bno codigo\b",
    r"\bno cÃ³digo\b",
    r"\bpelo codigo\b",
    r"\bpelo cÃ³digo\b",
    r"\bsource files?\b",
)


def _query_requests_code(query: str) -> bool:
    normalized_query = _normalize_for_match(query)
    return any(re.search(pattern, normalized_query) for pattern in CODE_REQUEST_PATTERNS)


def _query_requests_source_code(query: str) -> bool:
    normalized_query = _normalize_for_match(query)
    return any(re.search(pattern, normalized_query) for pattern in SOURCE_CODE_REQUEST_PATTERNS)


_CODE_LIKE_PROSE_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "after",
    "before",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "run",
    "start",
    "the",
    "this",
    "to",
    "open",
    "install",
    "use",
    "we",
    "with",
    "when",
    "then",
    "you",
    "your",
}


def _is_code_like_sentence(sentence: str) -> bool:
    normalized_sentence = sentence.strip()
    if not normalized_sentence:
        return False
    if "```" in normalized_sentence:
        return True
    if re.fullmatch(r"`[^`]+`", normalized_sentence):
        return True
    if re.search(r"[.!?]", normalized_sentence):
        return False
    if re.match(r"^(?:[$#]\s*)?(?:def|class|import|from|return|const|let|var|function|if|for|while|try|except|print|console\.log|npm|pip|git|python|bash|sh|node)\b", normalized_sentence):
        return True
    words = [word.lower() for word in re.findall(r"[A-Za-z_][\w.-]*", normalized_sentence)]
    prose_word_count = sum(1 for word in words if word in _CODE_LIKE_PROSE_WORDS)
    if prose_word_count > 1:
        return False
    if re.fullmatch(r"(?:[$#]\s*)?[A-Za-z0-9_./-]+(?:\s+[A-Za-z0-9_./-]+){0,6}", normalized_sentence):
        return True
    if re.search(r"[{}();<>]=?|=>", normalized_sentence) and len(words) <= 8:
        return True
    return False


def _is_mostly_code(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    code_lines = sum(1 for line in lines if _is_code_like_sentence(line))
    return code_lines / len(lines) >= 0.5


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
        "architecture": ("architecture", "arquitetura", "mvc", "layer", "layers", "camada", "camadas", "module", "modules", "modulo", "modulos", "estrutura", "fluxo", "controller", "controllers"),
        "setup": ("setup", "install", "instalar", "executar", "rodar", "ambiente", "env", "script", "scripts", "deploy", "deployment", "docker", "compose", "build"),
        "security": ("security", "seguranca", "auth", "authentication", "jwt", "senha", "password", "bcrypt", "token"),
        "evaluation": ("evaluation", "avaliacao", "avaliar", "metric", "metrics", "metrica", "metricas", "test", "tests"),
        "fine_tune": ("fine tune", "fine-tune", "fine tunning", "fine tuning", "finetuning", "training data", "hyperparameter", "lora", "qlora", "dataset"),
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
    if "autenticacao" in normalized or "autentic" in normalized:
        rewrite_parts.append("auth authentication service guard token")
    if "indicacao" in normalized or "referencia" in normalized or "referral" in normalized:
        rewrite_parts.append("referral referralCode referrer register registration service")
    if "registro" in normalized or "cadastrar" in normalized or "cadastro" in normalized:
        rewrite_parts.append("register registration create user service")
    if "token" in normalized and ("attached" in normalized or "request" in normalized or "api" in normalized):
        rewrite_parts.append("authorization bearer headers")
    if "token" in normalized and ("stored" in normalized or "storage" in normalized or "cache" in normalized):
        rewrite_parts.append("localStorage cache")

    layer_terms = _expand_architecture_layer_terms(_significant_terms(normalized))
    layer_expansion = sorted(layer_terms - _significant_terms(normalized))
    if layer_expansion:
        rewrite_parts.append(" ".join(layer_expansion))

    rewritten_query = " ".join(dict.fromkeys(" ".join(rewrite_parts).split()))
    terms = _expand_architecture_layer_terms(_significant_terms(query) | _significant_terms(rewritten_query))
    return QueryAnalysis(
        original_query=query,
        rewritten_query=rewritten_query,
        intents=intents,
        terms=terms,
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

    if "architecture" in analysis.intents and _is_code_evidence_source(source):
        priority += 0.6

    if _query_prefers_project_documentation(analysis.original_query, analysis) and _is_readme_source(source_key):
        priority += 0.7

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
        if "free-tier rag workspace" in text_key or "advanced rag" in text_key or "advanced-rag" in text_key:
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


def _is_readme_source(source: str) -> bool:
    return source.lower().replace("\\", "/").endswith("readme.md")


def _is_code_evidence_source(source: str) -> bool:
    source_key = source.lower().replace("\\", "/")
    name = Path(source_key).name
    if _is_readme_source(source_key):
        return False
    if name in {filename.lower() for filename in CODE_EVIDENCE_FILENAMES}:
        return True
    return Path(source_key).suffix.lower() in CODE_EVIDENCE_EXTENSIONS


def _is_test_source(source: str) -> bool:
    source_key = source.lower().replace("\\", "/")
    name = Path(source_key).name
    return (
        "/test/" in source_key
        or "/tests/" in source_key
        or ".spec." in name
        or ".test." in name
    )


PROJECT_DOCUMENTATION_INTENTS = {"stack", "overview", "architecture", "setup", "security", "evaluation", "fine_tune"}
PROJECT_DOCUMENTATION_TERMS = {
    "about",
    "architecture",
    "arquitetura",
    "build",
    "deploy",
    "deployment",
    "docs",
    "documentation",
    "documentacao",
    "documentado",
    "features",
    "ferramentas",
    "install",
    "license",
    "limitations",
    "mvc",
    "overview",
    "project",
    "projeto",
    "readme",
    "roadmap",
    "setup",
    "stack",
    "troubleshoot",
    "troubleshooting",
}


def _query_prefers_project_documentation(query: str, analysis: QueryAnalysis) -> bool:
    if _query_requests_source_code(query):
        return False
    if PROJECT_DOCUMENTATION_INTENTS & set(analysis.intents):
        return True
    return bool(PROJECT_DOCUMENTATION_TERMS & analysis.terms)


def _is_markdown_documentation_source(source: str) -> bool:
    source_key = source.lower().replace("\\", "/")
    return Path(source_key).suffix.lower() in {".md", ".rst", ".txt"} and not _is_readme_source(source_key)


def _query_needs_code_evidence(query: str, analysis: QueryAnalysis) -> bool:
    if _query_requests_source_code(query):
        return True
    if "stack" in analysis.intents:
        return False
    if any(intent in analysis.intents for intent in ("architecture", "security")):
        return True
    implementation_terms = {
        "api",
        "auth",
        "backend",
        "class",
        "component",
        "controller",
        "database",
        "entity",
        "flow",
        "frontend",
        "function",
        "guard",
        "hook",
        "implementation",
        "implements",
        "module",
        "referral",
        "referrer",
        "repository",
        "route",
        "service",
        "services",
        "source",
    }
    return bool(analysis.terms & implementation_terms)


def _node_key(item: Any) -> str:
    node = getattr(item, "node", item)
    return str(getattr(node, "node_id", "") or _source_doc(item))


def _source_scope_score(source: str, analysis: QueryAnalysis) -> float:
    source_key = source.lower().replace("\\", "/")
    query_key = _normalize_for_match(analysis.original_query)
    original_terms = _significant_terms(analysis.original_query)
    score = 0.0
    source_terms = _source_path_terms(source_key)
    score += 0.2 * len(source_terms & analysis.terms)
    if "architecture" in analysis.intents:
        for layer_terms in ARCHITECTURE_LAYER_GROUPS.values():
            if (analysis.terms & layer_terms) and (source_terms & layer_terms):
                score += 1.0
    if "frontend" in original_terms or "front" in original_terms:
        if "/frontend/" in source_key:
            score += 0.3
        if "/backend/" in source_key:
            score -= 0.8
    if "backend" in original_terms or "back" in original_terms:
        if "/backend/" in source_key:
            score += 0.3
        if "/frontend/" in source_key:
            score -= 0.8
    if {"api", "request", "requests"} & analysis.terms and ("api." in source_key or "/services/" in source_key):
        score += 1.0
    if {"login", "state"} & analysis.terms and ("authcontext" in source_key or "/contexts/" in source_key):
        score += 1.0
    if {"stored", "storage", "token"} & analysis.terms and ("/cache." in source_key or "/auth." in source_key):
        score += 1.0
    if {"stored", "storage", "localstorage", "token"} & analysis.terms and source_key.endswith("/utils/cache.ts"):
        score += 1.6
    if {"authorization", "bearer", "request", "requests", "api"} & analysis.terms and source_key.endswith("/services/api.ts"):
        score += 1.4
    if {"referral", "registration", "register"} & analysis.terms and ("auth.service" in source_key or "users.service" in source_key):
        score += 1.0
    if ("authservice" in query_key or "incrementreferrerscore" in query_key or {"register", "registration", "referralcode"} & analysis.terms) and source_key.endswith("/auth/auth.service.ts"):
        score += 2.4
    if ("authservice" in query_key or "incrementreferrerscore" in query_key) and "/frontend/" in source_key:
        score -= 2.5
    if {"register", "registration", "referralcode"} & analysis.terms and "/auth/guards/" in source_key:
        score -= 1.4
    if {"jwt", "public", "routes", "guard"} & original_terms and ("guard" in source_key or "decorator" in source_key or "controller" in source_key):
        score += 1.0
    if "jwt" in original_terms and "guard" in original_terms and source_key.endswith("/auth/guards/jwt-auth.guard.ts"):
        score += 2.0
    if "public" in original_terms and source_key.endswith("/auth/decorators/public.decorator.ts"):
        score += 1.0
    if {"database", "configuration", "configured"} & analysis.terms and source_key.endswith("/database/database.module.ts"):
        score += 2.4
    if "entity" in analysis.terms and source_key.endswith("/users/user.entity.ts"):
        score += 1.2
    if ("usersservice" in query_key or "getprofile" in query_key or {"profile", "referrallink"} & analysis.terms) and source_key.endswith("/users/users.service.ts"):
        score += 2.4
    if ("usersservice" in query_key or "getprofile" in query_key) and source_key.endswith("/auth/auth.service.ts"):
        score -= 1.0
    if "clear-database" in source_key and not ({"clear", "clean", "limpar", "script", "scripts"} & analysis.terms):
        score -= 2.0
    if "module" not in analysis.terms and ".module." in source_key:
        score -= 0.6
    if "database" not in analysis.terms and "/database/" in source_key:
        score -= 0.6
    return score


def _source_opposes_query_scope(source: str, analysis: QueryAnalysis) -> bool:
    source_key = source.lower().replace("\\", "/")
    original_terms = _significant_terms(analysis.original_query)
    asks_frontend = "frontend" in original_terms or "front" in original_terms
    asks_backend = "backend" in original_terms or "back" in original_terms
    return (asks_frontend and "/backend/" in source_key) or (asks_backend and "/frontend/" in source_key)


def _is_maintenance_source(source: str, analysis: QueryAnalysis) -> bool:
    source_key = source.lower().replace("\\", "/")
    if {"clear", "clean", "limpar", "setup", "script", "scripts"} & set(analysis.intents + list(analysis.terms)):
        return False
    return "clear-database" in source_key or "/scripts/" in source_key


def _is_manifest_source(source: str, analysis: QueryAnalysis) -> bool:
    source_key = source.lower().replace("\\", "/")
    if {"stack", "setup"} & set(analysis.intents):
        return False
    if {"package", "dependency", "dependencies", "script", "scripts"} & analysis.terms:
        return False
    return _is_dependency_manifest_source(source_key)


def _is_dependency_manifest_source(source: str) -> bool:
    source_key = source.lower().replace("\\", "/")
    return source_key.endswith(
        (
            "package.json",
            "requirements.txt",
            "pyproject.toml",
            "cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "gemfile",
            "composer.json",
            "pubspec.yaml",
            "mix.exs",
            "deno.json",
        )
    )


def _source_path(source: str) -> Path | None:
    path = Path(source)
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    return candidate if candidate.exists() and candidate.is_file() else None


def _focused_file_excerpt(source: str, analysis: QueryAnalysis, max_lines: int = 12) -> str | None:
    path = _source_path(source)
    if path is None:
        return None
    text = _safe_read_text(path)
    lines = [line.rstrip() for line in text.splitlines()]
    query_terms = analysis.terms
    ranked: list[tuple[float, int]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _is_low_value_code_evidence(stripped):
            continue
        normalized = _normalize_for_match(re.sub(r"(?<=[a-z])(?=[A-Z])|_", " ", stripped))
        terms = _significant_terms(normalized)
        overlap = len(query_terms & terms)
        if overlap <= 0:
            continue
        code_bonus = 0.4 if re.search(r"\b(async|await|return|throw|if|const|let|class|function|private|public|static)\b", stripped) else 0.0
        ranked.append((overlap + code_bonus, index))
    if not ranked:
        return None

    selected_indexes = sorted(index for _, index in sorted(ranked, key=lambda item: (-item[0], item[1]))[:4])
    excerpt_indexes: set[int] = set()
    for index in selected_indexes:
        excerpt_indexes.update(range(max(index - 1, 0), min(index + 2, len(lines))))
    excerpt_lines = [lines[index].strip() for index in sorted(excerpt_indexes) if lines[index].strip()]
    if not excerpt_lines:
        return None
    return f"Focused excerpt from {source}:\n" + "\n".join(excerpt_lines[:max_lines])


def _dedupe_results(results: Sequence[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        signature = (_source_doc(result), _snippet(_node_text(result), max_chars=180))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(result)
    return deduped


def _deepen_code_evidence_results(
    query: str,
    results: Sequence[Any],
    candidate_nodes: Sequence[Any],
    top_k: int,
) -> tuple[list[Any], dict[str, Any]]:
    analysis = analyze_query(query)
    trace = {
        "enabled": False,
        "added_sources": [],
        "reason": None,
    }
    allow_tests = "evaluation" in analysis.intents
    preserve_project_docs = _query_prefers_project_documentation(query, analysis)
    selected = _dedupe_results(results)
    if not _query_needs_code_evidence(query, analysis) or not candidate_nodes:
        return selected, trace

    scoped_selected = [result for result in selected if not _source_opposes_query_scope(_source_doc(result), analysis)]
    if scoped_selected:
        selected = scoped_selected
    if not allow_tests:
        non_test_selected = [result for result in selected if not _is_test_source(_source_doc(result))]
        if non_test_selected:
            selected = non_test_selected
    non_manifest_selected = [result for result in selected if not _is_manifest_source(_source_doc(result), analysis)]
    if non_manifest_selected:
        selected = non_manifest_selected
    selected_keys = {_node_key(result) for result in selected}
    if _query_requests_code(query):
        wanted_code_sources = 3
    elif "architecture" in analysis.intents:
        wanted_code_sources = max(2, min(3, _requested_architecture_layer_count(analysis)))
    elif "security" in analysis.intents:
        wanted_code_sources = 2
    else:
        wanted_code_sources = 1

    candidates_by_source: dict[str, LocalNodeWithScore] = {}
    for node in candidate_nodes:
        if _node_key(node) in selected_keys:
            continue
        source = _source_doc(node)
        if not _is_code_evidence_source(source):
            continue
        if _source_opposes_query_scope(source, analysis) or _is_maintenance_source(source, analysis) or _is_manifest_source(source, analysis):
            continue
        if _is_test_source(source) and not allow_tests:
            continue
        score = (
            _score_local_node(node, analysis)
            + (0.5 * _lexical_score(analysis.rewritten_query, source))
            + _source_scope_score(source, analysis)
        )
        if score <= 0:
            continue
        candidate = LocalNodeWithScore(node=getattr(node, "node", node), score=score)
        if source not in candidates_by_source or score > (candidates_by_source[source].score or 0):
            candidates_by_source[source] = candidate

    for source, candidate in list(candidates_by_source.items()):
        excerpt = _focused_file_excerpt(source, analysis)
        if not excerpt:
            continue
        focused_node = LocalTextNode(
            text=excerpt,
            node_id=f"{source}#focused",
            metadata={"file_name": source, "source": "focused_file_excerpt"},
        )
        candidates_by_source[source] = LocalNodeWithScore(
            node=focused_node,
            score=(candidate.score or 0) + 1.0,
        )

    candidates = list(candidates_by_source.values())
    candidates.sort(key=lambda item: (item.score, _is_test_source(_source_doc(item)), _source_doc(item)), reverse=True)
    added_sources: list[str] = []
    for candidate in candidates:
        selected_code_sources = {
            _source_doc(result)
            for result in selected
            if _is_code_evidence_source(_source_doc(result))
            and (allow_tests or not _is_test_source(_source_doc(result)))
        }
        selected_top_candidate_sources = selected_code_sources & { _source_doc(item) for item in candidates[:wanted_code_sources] }
        if len(selected_top_candidate_sources) >= min(wanted_code_sources, len(candidates)):
            break
        source = _source_doc(candidate)
        if source in added_sources:
            continue
        if source in selected_code_sources:
            existing_index = next((index for index, result in enumerate(selected) if _source_doc(result) == source), None)
            if existing_index is not None:
                existing_score = (
                    _score_local_node(selected[existing_index], analysis)
                    + (0.5 * _lexical_score(analysis.rewritten_query, source))
                    + _source_scope_score(source, analysis)
                )
                if (candidate.score or 0) > existing_score:
                    selected[existing_index] = candidate
                    added_sources.append(source)
            continue
        if len(selected) >= top_k:
            if preserve_project_docs:
                removable_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if not _is_readme_source(_source_doc(selected[index]))
                        and not _is_dependency_manifest_source(_source_doc(selected[index]))
                        and (_is_test_source(_source_doc(selected[index])) or not preserve_project_docs)
                    ),
                    next(
                        (
                            index
                            for index in range(len(selected) - 1, -1, -1)
                            if not _is_readme_source(_source_doc(selected[index]))
                            and not _is_dependency_manifest_source(_source_doc(selected[index]))
                        ),
                        len(selected) - 1,
                    ),
                )
            else:
                removable_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if _is_readme_source(_source_doc(selected[index])) or (_is_test_source(_source_doc(selected[index])) and not allow_tests)
                    ),
                    len(selected) - 1,
                )
            selected.pop(removable_index)
        selected.append(candidate)
        selected_keys.add(_node_key(candidate))
        added_sources.append(source)

    if added_sources:
        selected.sort(key=lambda item: (_score(item) if _score(item) is not None else 0.0), reverse=True)
        trace.update(
            {
                "enabled": True,
                "added_sources": added_sources,
                "reason": "code_evidence_diversity",
            }
        )
    return selected, trace


def _project_evidence_group(source: str, analysis: QueryAnalysis) -> int:
    if _is_readme_source(source):
        return 0
    if _is_dependency_manifest_source(source):
        return 1
    if _is_markdown_documentation_source(source):
        return 2
    if _is_code_evidence_source(source):
        return 3
    return 4


def _balance_project_evidence_results(
    query: str,
    results: Sequence[Any],
    candidate_nodes: Sequence[Any],
    top_k: int,
) -> tuple[list[Any], dict[str, Any]]:
    analysis = analyze_query(query)
    trace = {
        "enabled": False,
        "added_sources": [],
        "reason": None,
    }
    if not _query_prefers_project_documentation(query, analysis) or not candidate_nodes:
        return list(results), trace

    selected = _dedupe_results(results)
    selected_keys = {_node_key(result) for result in selected}
    selected_sources = {_source_doc(result) for result in selected}

    def candidate_allowed(source: str) -> bool:
        if _source_opposes_query_scope(source, analysis):
            return False
        if _is_maintenance_source(source, analysis):
            return False
        if _is_test_source(source) and "evaluation" not in analysis.intents:
            return False
        return (
            _is_readme_source(source)
            or _is_markdown_documentation_source(source)
            or _is_dependency_manifest_source(source)
            or (_query_needs_code_evidence(query, analysis) and _is_code_evidence_source(source))
        )

    candidates: list[LocalNodeWithScore] = []
    for node in candidate_nodes:
        if _node_key(node) in selected_keys:
            continue
        source = _source_doc(node)
        if source in selected_sources or not candidate_allowed(source):
            continue
        text = _node_text(node)
        score = _score_local_node(node, analysis)
        if _is_readme_source(source):
            score += 1.5
        elif _is_dependency_manifest_source(source):
            score += 0.6 if {"stack", "setup"} & set(analysis.intents) else 0.2
        elif _is_markdown_documentation_source(source):
            score += 0.4
        elif _is_code_evidence_source(source):
            score += _source_scope_score(source, analysis)
        if score <= 0:
            continue
        candidates.append(LocalNodeWithScore(node=getattr(node, "node", node), score=score))

    def project_rank(item: Any) -> tuple[int, float, str]:
        source = _source_doc(item)
        return (_project_evidence_group(source, analysis), -float(_score(item) or 0.0), source)

    def removable_index() -> int:
        return next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if not _is_readme_source(_source_doc(selected[index]))
                and not _is_dependency_manifest_source(_source_doc(selected[index]))
            ),
            len(selected) - 1,
        )

    added_sources: list[str] = []
    for candidate in sorted(candidates, key=project_rank):
        source = _source_doc(candidate)
        if source in {_source_doc(item) for item in selected}:
            continue
        group = _project_evidence_group(source, analysis)
        if group == 3 and any(
            _project_evidence_group(_source_doc(item), analysis) == 3
            and _architecture_layer_bucket(_source_doc(item)) == _architecture_layer_bucket(source)
            for item in selected
        ):
            continue
        if len(selected) >= top_k:
            selected.pop(removable_index())
        selected.append(candidate)
        added_sources.append(source)

        has_readme = any(_is_readme_source(_source_doc(item)) for item in selected)
        support_count = sum(
            1
            for item in selected
            if not _is_readme_source(_source_doc(item))
            and (
                _is_dependency_manifest_source(_source_doc(item))
                or _is_markdown_documentation_source(_source_doc(item))
                or _is_code_evidence_source(_source_doc(item))
            )
        )
        if has_readme and support_count >= 2:
            break

    if added_sources:
        selected = sorted(selected, key=project_rank)[:top_k]
        trace.update(
            {
                "enabled": True,
                "added_sources": added_sources,
                "reason": "project_documentation_and_source_evidence",
            }
        )
    return selected, trace


def _balance_stack_evidence_results(
    query: str,
    results: Sequence[Any],
    candidate_nodes: Sequence[Any],
    top_k: int,
) -> tuple[list[Any], dict[str, Any]]:
    analysis = analyze_query(query)
    trace = {
        "enabled": False,
        "added_sources": [],
        "reason": None,
    }
    if "stack" not in analysis.intents or not candidate_nodes:
        return list(results), trace

    selected = _dedupe_results(results)
    selected_keys = {_node_key(result) for result in selected}
    asks_frontend = "frontend" in analysis.terms or "front" in analysis.terms
    asks_backend = "backend" in analysis.terms or "back" in analysis.terms

    def in_scope(source: str) -> bool:
        source_key = source.lower().replace("\\", "/")
        if asks_frontend and "/backend/" in source_key:
            return False
        if asks_backend and "/frontend/" in source_key:
            return False
        return True

    candidates: list[LocalNodeWithScore] = []
    for node in candidate_nodes:
        if _node_key(node) in selected_keys:
            continue
        source = _source_doc(node)
        if not in_scope(source):
            continue
        source_key = source.lower().replace("\\", "/")
        is_readme = _is_readme_source(source)
        is_manifest = _is_dependency_manifest_source(source)
        if not is_readme and not is_manifest:
            continue
        text = _node_text(node)
        score = _score_local_node(node, analysis)
        if is_readme and ("stack" in _normalize_for_match(text) or "ferramentas" in _normalize_for_match(text)):
            score += 2.0
        if is_manifest:
            score += 1.5
        if asks_frontend and "/frontend/" in source_key:
            score += 1.0
        if asks_backend and "/backend/" in source_key:
            score += 1.0
        if score <= 0:
            continue
        candidates.append(LocalNodeWithScore(node=getattr(node, "node", node), score=score))

    def stack_rank(item: Any) -> tuple[int, float, str]:
        source = _source_doc(item)
        if _is_readme_source(source):
            group = 0
        elif _is_dependency_manifest_source(source):
            group = 1
        elif _is_code_evidence_source(source):
            group = 2
        else:
            group = 3
        return (group, -float(_score(item) or 0.0), source)

    added_sources: list[str] = []
    for candidate in sorted(candidates, key=stack_rank):
        source = _source_doc(candidate)
        if source in {_source_doc(item) for item in selected}:
            continue
        if len(selected) >= top_k:
            removable_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if not _is_readme_source(_source_doc(selected[index]))
                    and not _is_dependency_manifest_source(_source_doc(selected[index]))
                ),
                len(selected) - 1,
            )
            selected.pop(removable_index)
        selected.append(candidate)
        added_sources.append(source)

    if added_sources:
        selected = sorted(selected, key=stack_rank)[:top_k]
        trace.update(
            {
                "enabled": True,
                "added_sources": added_sources,
                "reason": "stack_declared_and_confirmed_evidence",
            }
        )
    return selected, trace


def _local_retrieval_score(query: str, text: str, source: str) -> float:
    analysis = analyze_query(query)
    lexical = _lexical_score(analysis.rewritten_query, text)
    return lexical + _source_priority(source, text, analysis)


def _score_local_node(node: Any, analysis: QueryAnalysis) -> float:
    text = _node_text(node)
    return _lexical_score(analysis.rewritten_query, text) + _source_priority(_source_doc(node), text, analysis)


def _chunk_text(text: str, max_chars: int = 2400) -> list[str]:
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
        framework_hints.append("SQLite3")
    if "@nestjs/jwt" in all_dependency_names or "jsonwebtoken" in all_dependency_names:
        framework_hints.append("JWT")
    if "class-validator" in all_dependency_names:
        framework_hints.append("class-validator")
    if "class-transformer" in all_dependency_names:
        framework_hints.append("class-transformer")
    if "bcrypt" in all_dependency_names:
        framework_hints.append("bcrypt")
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


def _is_noise_chunk(text: str) -> bool:
    """Filter chunks that are code blocks, not documentation."""
    stripped = text.strip()
    if stripped.startswith("```") or stripped.startswith(">>>"):
        return True
    first_lines = stripped.split("\n")[:3]
    for line in first_lines:
        ls = line.lstrip()
        # Shell commands at column 0
        if ls.startswith(("$", "#>", "- ", "> ")) and any(ls.startswith(p) for p in ("$ pip", "$ npm", "# pip", "- pip", "$ node", "$ python")):
            return True
        if ls.startswith("if __name__"):
            return True
        if any(ls.startswith(kw) for kw in ("import ", "from ", "def ", "class ", "return ", "result =", "=> ")):
            return True
        if ls.startswith("```"):
            return True
    return False


@dataclass
class FineTuneMetadata:
    """Metadata extracted from HuggingFace model cards."""
    base_model: str | None = None
    dataset: str | list[str] | None = None
    training_details: dict[str, Any] | None = None
    evaluation_metrics: dict[str, Any] | None = None
    lora_config: dict[str, Any] | None = None
    license: str | None = None
    language: list[str] | None = None
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "dataset": self.dataset,
            "training_details": self.training_details,
            "evaluation_metrics": self.evaluation_metrics,
            "lora_config": self.lora_config,
            "license": self.license,
            "language": self.language,
            "tags": self.tags,
        }

    def to_summary(self) -> str:
        parts: list[str] = []
        if self.base_model:
            parts.append(f"Base Model: {self.base_model}")
        if self.dataset:
            if isinstance(self.dataset, list):
                parts.append(f"Dataset: {', '.join(self.dataset)}")
            else:
                parts.append(f"Dataset: {self.dataset}")
        if self.training_details:
            details = []
            for key, value in self.training_details.items():
                details.append(f"{key}: {value}")
            if details:
                parts.append(f"Training: {', '.join(details)}")
        if self.evaluation_metrics:
            metrics = [f"{key}: {value}" for key, value in self.evaluation_metrics.items()]
            if metrics:
                parts.append(f"Evaluation: {', '.join(metrics)}")
        if self.lora_config:
            lora_parts = [f"{key}: {value}" for key, value in self.lora_config.items()]
            if lora_parts:
                parts.append(f"LoRA: {', '.join(lora_parts)}")
        return " | ".join(parts) if parts else None


def _fine_tune_has_brief_fields(metadata: FineTuneMetadata) -> bool:
    return any(
        (
            metadata.base_model,
            metadata.dataset,
            metadata.training_details,
            metadata.evaluation_metrics,
        )
    )


def _format_fine_tune_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_fine_tune_mapping(values: dict[str, Any], key_overrides: dict[str, str] | None = None) -> str:
    key_overrides = key_overrides or {}
    return ", ".join(
        f"{key_overrides.get(key, key)}={_format_fine_tune_value(value)}"
        for key, value in values.items()
    )


def _format_fine_tune_extract_answer(metadata: FineTuneMetadata) -> str | None:
    if not _fine_tune_has_brief_fields(metadata):
        return None

    dataset = metadata.dataset
    if isinstance(dataset, list):
        dataset_text = ", ".join(dataset) if dataset else "Unknown"
    else:
        dataset_text = dataset or "Unknown"

    training_text = "Unknown"
    if metadata.training_details:
        training_text = _format_fine_tune_mapping(metadata.training_details, {"Alpha": "alpha"})

    metrics_text = "Unknown"
    if metadata.evaluation_metrics:
        metrics_text = _format_fine_tune_mapping(metadata.evaluation_metrics)

    unknowns = []
    if not metadata.base_model:
        unknowns.append("Base model")
    if not metadata.dataset:
        unknowns.append("Dataset")
    if not metadata.training_details:
        unknowns.append("Training")
    if not metadata.evaluation_metrics:
        unknowns.append("Metrics")

    return "\n".join(
        (
            f"Base model: {metadata.base_model or 'Unknown'}",
            f"Dataset: {dataset_text}",
            f"Training: {training_text}",
            f"Metrics: {metrics_text}",
            f"Unknowns: {', '.join(unknowns) if unknowns else 'None'}",
        )
    )


def _parse_yaml_frontmatter(text: str) -> dict[str, Any]:
    """Parse YAML frontmatter from HuggingFace README.md.

    Returns dict of parsed values or empty dict if no frontmatter.
    Does not require external YAML library - uses simple line-by-line parsing.
    """
    lines = text.strip().split("\n")
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == "---")
    except StopIteration:
        return {}

    yaml_lines: list[str] = []

    for line in lines[start + 1:]:
        if line.strip() == "---":
            break
        yaml_lines.append(line)

    if not yaml_lines:
        return {}

    result: dict[str, Any] = {}
    current_key: str | None = None
    current_indent = 0
    in_multiline = False
    multiline_value: list[str] = []

    for line in yaml_lines:
        if not line.strip():
            continue

        if line.startswith("  ") or line.startswith("- "):
            if current_key:
                existing = result.get(current_key)
                if isinstance(existing, list):
                    item = line.strip()
                    if item.startswith("- "):
                        existing.append(item[2:].strip())
                    else:
                        existing.append(item)
            continue

        if ":" in line:
            key_part, value_part = line.split(":", 1)
            key = key_part.strip()
            value = value_part.strip()

            if value == "":
                result[key] = []
                current_key = key
            elif value.startswith("[") and value.endswith("]"):
                items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
                result[key] = items
                current_key = key
            elif value.startswith("-"):
                result[key] = [value[1:].strip()]
                current_key = key
            else:
                result[key] = value
                current_key = key

    return result


def _parse_markdown_section(text: str, section_name: str) -> str:
    """Extract content under a specific markdown section heading."""
    pattern = rf"^##\s+{re.escape(section_name)}\s*$"
    lines = text.split("\n")
    start_idx = None

    for i, line in enumerate(lines):
        if re.match(pattern, line, re.IGNORECASE):
            start_idx = i + 1
            break

    if start_idx is None:
        return ""

    content_lines: list[str] = []
    for line in lines[start_idx:]:
        if re.match(r"^##\s+", line):
            break
        content_lines.append(line)

    return "\n".join(content_lines).strip()


def _extract_fine_tune_info(contexts: Sequence[str], sources: Sequence[dict[str, Any]]) -> FineTuneMetadata:
    """Extract structured fine-tuning metadata from HuggingFace model cards.

    Parses:
    - YAML frontmatter for base_model, datasets, license, tags, language
    - Training Details section for epochs, batch_size, learning_rate, etc.
    - Evaluation section for metrics
    - LoRA/QLoRA config tables

    Args:
        contexts: List of retrieved text chunks
        sources: List of source metadata dicts

    Returns:
        FineTuneMetadata with extracted fields (None for missing fields)
    """
    metadata = FineTuneMetadata()

    for context, source in zip(contexts, sources):
        source_doc = source.get("source_doc", "")
        if "README.md" not in source_doc:
            continue

        yaml_data = _parse_yaml_frontmatter(context)

        if yaml_data.get("base_model"):
            metadata.base_model = yaml_data["base_model"]
        if yaml_data.get("datasets"):
            datasets = yaml_data["datasets"]
            metadata.dataset = datasets if isinstance(datasets, list) else [datasets]
        if yaml_data.get("license"):
            metadata.license = yaml_data["license"]
        if yaml_data.get("language"):
            langs = yaml_data["language"]
            metadata.language = langs if isinstance(langs, list) else [langs]
        if yaml_data.get("tags"):
            tags = yaml_data["tags"]
            metadata.tags = tags if isinstance(tags, list) else [tags]

        training_text = _parse_markdown_section(context, "Training Details")
        if training_text:
            details: dict[str, Any] = {}
            for line in training_text.split("\n"):
                if ":" in line and not line.startswith("|"):
                    key, value = line.split(":", 1)
                    key = key.strip().replace("*", "").strip()
                    value = value.strip()
                    try:
                        if "." in value:
                            details[key] = float(value)
                        else:
                            details[key] = int(value)
                    except ValueError:
                        details[key] = value
                elif "|" in line and line.strip().startswith("|"):
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2:
                        key = parts[0].replace("*", "").strip()
                        value = parts[1].strip()
                        try:
                            if "." in value:
                                details[key] = float(value)
                            else:
                                details[key] = int(value)
                        except ValueError:
                            details[key] = value
            if details:
                metadata.training_details = details

        eval_text = _parse_markdown_section(context, "Evaluation")
        if eval_text:
            metrics: dict[str, Any] = {}
            table_headers: list[str] = []
            for line in eval_text.split("\n"):
                if "|" not in line:
                    continue
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if not parts or all(set(part) <= {"-", ":"} for part in parts):
                    continue
                if any("exact match" in part.lower() for part in parts):
                    table_headers = parts
                    continue
                if table_headers:
                    for header, value in zip(table_headers, parts):
                        if "exact match" not in header.lower():
                            continue
                        match = re.search(r"(\d+\.?\d*)%?", value)
                        if match:
                            metrics["exact_match"] = float(match.group(1))
            if metrics:
                metadata.evaluation_metrics = metrics

        lora_text = _parse_markdown_section(context, "LoRA Config")
        if lora_text:
            lora: dict[str, Any] = {}
            for line in lora_text.split("\n"):
                if "|" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2:
                        key = parts[0].replace("*", "").strip().title()
                        value = parts[1].strip()
                        try:
                            lora[key] = int(value)
                        except ValueError:
                            try:
                                lora[key] = float(value)
                            except ValueError:
                                lora[key] = value
            if lora:
                metadata.lora_config = lora

    return metadata


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
    if _looks_like_json_text(context) and not context.lstrip().startswith("JSON document summary"):
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
    frontend_tech = {"React Router DOM", "React Router", "TypeScript", "React", "Vite", "Axios", "ESLint", "Jest", "Prettier"}
    backend_tech = {"NestJS", "Express", "TypeORM", "SQLite3", "SQLite", "JWT", "class-validator", "class-transformer", "bcrypt"}
    found: list[str] = []
    for tech in KNOWN_TECHNOLOGIES:
        if frontend_only and tech in backend_tech:
            continue
        if backend_only and tech in frontend_tech:
            continue
        if _normalize_for_match(tech) in normalized:
            found.append(tech)
    return found


STACK_GROUPS = {
    "Backend": (
        "NestJS",
        "Express",
        "TypeScript",
        "TypeORM",
        "SQLite3",
        "SQLite",
        "JWT",
        "class-validator",
        "class-transformer",
        "bcrypt",
    ),
    "Frontend": (
        "React",
        "TypeScript",
        "Vite",
        "React Router DOM",
        "React Router",
        "Axios",
    ),
    "Qualidade": (
        "ESLint",
        "Jest",
        "Prettier",
    ),
}


def _context_mentions_tech(context: str, tech: str) -> bool:
    normalized = _normalize_for_match(context)
    if _normalize_for_match(tech) in normalized:
        return True
    aliases = {
        "JWT": ("@nestjs/jwt", "jsonwebtoken"),
        "SQLite3": ("sqlite3", "sqlite"),
        "React Router DOM": ("react-router-dom", "react router"),
        "class-validator": ("class-validator", "class validator"),
        "class-transformer": ("class-transformer", "class transformer"),
    }
    return any(_normalize_for_match(alias) in normalized for alias in aliases.get(tech, ()))


def _stack_group_for_context(source: str, text: str) -> str | None:
    source_key = source.lower().replace("\\", "/")
    text_key = _normalize_for_match(text)
    if _is_readme_source(source_key):
        return None
    if "/backend/" in source_key or "### backend" in text_key or "backend package" in text_key:
        return "Backend"
    if "/frontend/" in source_key or "### frontend" in text_key or "frontend package" in text_key:
        return "Frontend"
    if "### qualidade" in text_key or "quality" in text_key:
        return "Qualidade"
    return None


def _format_stack_answer(
    contexts: Sequence[str],
    sources: Sequence[dict[str, Any]],
    query: str = "",
) -> str | None:
    query_terms = _significant_terms(query)
    frontend_only = "frontend" in query_terms or "front" in query_terms
    backend_only = "backend" in query_terms or "back" in query_terms
    declared: dict[str, set[str]] = {group: set() for group in STACK_GROUPS}
    confirmed: dict[str, dict[str, set[str]]] = {group: {} for group in STACK_GROUPS}

    for index, context in enumerate(contexts):
        source = str(sources[index].get("source_doc", f"document_{index + 1}")) if index < len(sources) else f"document_{index + 1}"
        is_readme = _is_readme_source(source)
        context_group = _stack_group_for_context(source, context)
        for group, technologies in STACK_GROUPS.items():
            if frontend_only and group == "Backend":
                continue
            if backend_only and group == "Frontend":
                continue
            if context_group and context_group != group and not (group == "Qualidade" and any(_context_mentions_tech(context, tech) for tech in technologies)):
                continue
            for tech in technologies:
                if _context_mentions_tech(context, tech):
                    if is_readme:
                        declared[group].add(tech)
                    else:
                        confirmed[group].setdefault(tech, set()).add(source)

    has_declared = any(declared[group] for group in declared)
    if not has_declared:
        for group, tech_sources in confirmed.items():
            declared[group].update(tech_sources)

    lines = ["Stack do projeto:"]
    wrote_group = False
    for group, technologies in STACK_GROUPS.items():
        techs = [tech for tech in technologies if tech in declared[group] or tech in confirmed[group]]
        if not techs:
            continue
        wrote_group = True
        lines.append(f"{group}:")
        for tech in techs:
            lines.append(f"- {tech}")
    return "\n".join(lines) if wrote_group else None




class ChatLLMClient:
    def __init__(self, policy: ChatProviderPolicy) -> None:
        import cloud_ragas

        self.cache_enabled = policy.cache_enabled
        self.client = cloud_ragas.FreeTierCloudClient(
            budget=cloud_ragas.CloudCallBudget(max_calls=policy.max_calls),
            providers=policy.providers,
        )

    def generate_text(self, prompt: str, temperature: float | None = None) -> str:
        return self.client.generate_text(prompt, temperature=temperature)


def _get_chat_llm_client(policy: ChatProviderPolicy) -> Any:
    return ChatLLMClient(policy)


def _chat_cloud_error(code: str, retryable: bool, stage: str = "policy", provider: str | None = None) -> SynthesisError:
    return SynthesisError(code=code, stage=stage, retryable=retryable, provider=provider)


def _provider_attempts(policy: ChatProviderPolicy, *, outcome: str, error_class: str | None = None) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for index, provider in enumerate(policy.providers, start=1):
        attempt = {
            "provider": provider.name,
            "model": getattr(provider, "model", None),
            "attempt": index,
            "outcome": outcome,
        }
        if error_class:
            attempt["error_class"] = error_class
        attempts.append(attempt)
    return attempts


def _extractive_synthesis_result(
    query: str,
    contexts: Sequence[str],
    sources: Sequence[dict[str, Any]],
    error: SynthesisError,
    policy: ChatProviderPolicy | None = None,
    provider_attempts: list[dict[str, Any]] | None = None,
    fine_tune_metadata: FineTuneMetadata | None = None,
    intent: str = "general",
) -> SynthesisResult:
    answer = _format_fine_tune_extract_answer(fine_tune_metadata) if fine_tune_metadata else None
    if answer is None and intent == "stack":
        answer = _format_stack_answer(contexts, sources, query)
        if answer is None:
            techs = _extract_technologies(contexts, query)
            answer = ", ".join(techs) if techs else None
    if answer is None and _query_needs_code_evidence(query, analyze_query(query)):
        answer = synthesize_source_grounded_answer(query, contexts, sources)
    return SynthesisResult(
        answer=answer or synthesize_extractive_answer(query, contexts, max_sentences=5),
        mode="extractive",
        error=error,
        provider_chain=policy.provider_chain() if policy else [],
        provider_timeout_seconds=policy.provider_timeout_seconds if policy else None,
        total_timeout_seconds=policy.total_timeout_seconds if policy else None,
        provider_attempts=provider_attempts or [],
    )


def _generate_text_with_timeout(client: Any, prompt: str, policy: ChatProviderPolicy) -> str:
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.generate_text, prompt, temperature=0.2)
        per_provider_timeout = max(policy.provider_timeout_seconds, 0.0)
        total_timeout = max(policy.total_timeout_seconds, 0.0)

        if total_timeout == 0.0 or per_provider_timeout == 0.0:
            future.cancel()
            raise TimeoutError("chat timeout exceeded")

        timeout = min(per_provider_timeout, total_timeout)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("chat timeout exceeded") from exc

    if time.monotonic() - start > total_timeout:
        raise TimeoutError("chat timeout exceeded")
    return result


def synthesize_chat_answer(
    query: str,
    contexts: Sequence[str],
    intent: str,
    sources: Sequence[dict[str, Any]],
    fine_tune_metadata: FineTuneMetadata | None = None,
) -> SynthesisResult:
    if not contexts:
        error = SynthesisError(code=SYNTHESIS_NO_CONTEXTS, stage="precheck", retryable=False, provider=None)
        return SynthesisResult(answer=EMPTY_LOCAL_CONTEXT_ANSWER, mode="extractive", error=error)
    if not _has_enough_evidence(query, contexts, sources):
        error = SynthesisError(code=SYNTHESIS_INSUFFICIENT_EVIDENCE, stage="precheck", retryable=False, provider=None)
        return SynthesisResult(answer=LOW_EVIDENCE_ANSWER, mode="extractive", error=error)
    if intent == "stack":
        stack_answer = _format_stack_answer(contexts, sources, query)
        if stack_answer:
            return SynthesisResult(answer=stack_answer, mode="extractive")

    policy = ChatProviderPolicy.from_env()
    if not policy.enabled:
        return _extractive_synthesis_result(
            query,
            contexts,
            sources,
            _chat_cloud_error(SYNTHESIS_CLOUD_CHAT_DISABLED, retryable=False),
            policy,
            fine_tune_metadata=fine_tune_metadata,
            intent=intent,
        )
    if not policy.providers:
        return _extractive_synthesis_result(
            query,
            contexts,
            sources,
            _chat_cloud_error(SYNTHESIS_NO_PROVIDER_CONFIGURED, retryable=False),
            policy,
            fine_tune_metadata=fine_tune_metadata,
            intent=intent,
        )
    if policy.max_calls <= 0:
        return _extractive_synthesis_result(
            query,
            contexts,
            sources,
            _chat_cloud_error(SYNTHESIS_BUDGET_EXCEEDED, retryable=False),
            policy,
            fine_tune_metadata=fine_tune_metadata,
            intent=intent,
        )

    provider_attempts = _provider_attempts(policy, outcome="started")

    from synthesis import _build_prompt, _post_process_llm_response  # noqa: PLC0415

    prompt = _build_prompt(query, contexts, sources, intent=intent, fine_tune_metadata=fine_tune_metadata)

    try:
        raw_answer = _generate_text_with_timeout(_get_chat_llm_client(policy), prompt, policy)
    except TimeoutError:
        return _extractive_synthesis_result(
            query,
            contexts,
            sources,
            _chat_cloud_error(SYNTHESIS_PROVIDER_TIMEOUT, retryable=True, stage="provider"),
            policy,
            provider_attempts=_provider_attempts(policy, outcome="timeout", error_class="TimeoutError"),
            fine_tune_metadata=fine_tune_metadata,
            intent=intent,
        )
    except Exception as exc:
        return _extractive_synthesis_result(
            query,
            contexts,
            sources,
            _chat_cloud_error(SYNTHESIS_PROVIDER_EXHAUSTED, retryable=True, stage="provider"),
            policy,
            provider_attempts=_provider_attempts(policy, outcome="error", error_class=type(exc).__name__),
            fine_tune_metadata=fine_tune_metadata,
            intent=intent,
        )

    answer = _post_process_llm_response(raw_answer)
    if answer is None:
        return _extractive_synthesis_result(
            query,
            contexts,
            sources,
            _chat_cloud_error(SYNTHESIS_PROVIDER_EXHAUSTED, retryable=False, stage="provider"),
            policy,
            provider_attempts=_provider_attempts(policy, outcome="empty_response"),
            fine_tune_metadata=fine_tune_metadata,
            intent=intent,
        )
    return SynthesisResult(
        answer=answer,
        mode="generative",
        provider_chain=policy.provider_chain(),
        provider_timeout_seconds=policy.provider_timeout_seconds,
        total_timeout_seconds=policy.total_timeout_seconds,
        provider_attempts=_provider_attempts(policy, outcome="success"),
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
        return [c for c in chunks if c and not _is_noise_chunk(c)]
    text = _context_text_for_file(path).strip()
    return _chunk_text(text) if text else []


def _is_supported_local_context_file(path: Path) -> bool:
    return (
        path.name not in LOCAL_CONTEXT_IGNORED_FILENAMES
        and path.is_file()
        and (path.suffix.lower() in LOCAL_CONTEXT_EXTENSIONS or path.name in LOCAL_CONTEXT_FILENAMES)
    ) and not any(part in LOCAL_CONTEXT_IGNORED_DIR_NAMES for part in path.parts)


def _current_source_raw_path(raw_path: Path) -> tuple[bool, Path | None]:
    current_source_path = PROJECT_ROOT / "data" / "current_source.json"
    try:
        current_source = json.loads(current_source_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False, None
    source_slug = current_source.get("source_slug") if isinstance(current_source, dict) else None
    if not source_slug:
        return False, None
    if not raw_path.exists():
        return True, None
    for child in raw_path.iterdir():
        if child.is_dir() and child.name.lower() == str(source_slug).lower():
            return True, child
    return True, None


def _scoped_local_context_paths(paths: Sequence[Path]) -> list[Path]:
    scoped: list[Path] = []
    default_raw_path = PROJECT_ROOT / "data" / "raw"
    for path in paths:
        if path == default_raw_path:
            has_current_source, source_path = _current_source_raw_path(path)
            if source_path is not None:
                scoped.append(source_path)
            elif not has_current_source:
                scoped.append(path)
        else:
            scoped.append(path)
    return scoped


def load_local_context_nodes(paths: Sequence[Path] | None = None) -> list[LocalTextNode]:
    paths = list(paths) if paths is not None else _scoped_local_context_paths(LOCAL_CONTEXT_PATHS)
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

    from synthesis import _strip_wrapping_code_fences  # noqa: PLC0415

    query_terms = _tokenize(query)
    request_code = _query_requests_code(query)
    cleaned_contexts = [_context_for_synthesis(context) for context in contexts]
    code_heavy_context = bool(cleaned_contexts) and sum(
        1 for context in cleaned_contexts if _is_mostly_code(context)
    ) > len(cleaned_contexts) / 2
    should_include_code = request_code or code_heavy_context
    ranked: list[tuple[float, int, str]] = []

    for context_index, context in enumerate(cleaned_contexts):
        if not should_include_code:
            context = re.sub(r"```[\w-]*\n[\s\S]*?\n?```", " ", context, flags=re.MULTILINE)
            context = context.replace("```", " ").replace("`", " ")
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", context) if part.strip()]
        for sentence_index, sentence in enumerate(sentences or [context.strip()]):
            if not should_include_code and _is_code_like_sentence(sentence):
                continue
            terms = _tokenize(sentence)
            overlap = len(query_terms & terms)
            density = overlap / max(len(terms), 1)
            ranked.append((overlap + density, context_index * 1000 + sentence_index, sentence))

    if not ranked:
        if should_include_code:
            return _strip_wrapping_code_fences(contexts[0][:500]) if contexts else "No relevant context was retrieved."
        return "No non-code evidence was retrieved for this question. Ask for code if you want code examples."

    if not should_include_code:
        ranked = [item for item in ranked if not _is_code_like_sentence(item[2])]
        if not ranked:
            return "No non-code evidence was retrieved for this question. Ask for code if you want code examples."

    positive = [item for item in ranked if item[0] > 0]
    if positive:
        ranked = positive

    json_contexts = [context for context in cleaned_contexts if context.startswith("JSON document summary")]
    if json_contexts:
        return _strip_wrapping_code_fences(json_contexts[0][:500])

    best = sorted(ranked, key=lambda item: (-item[0], item[1]))[:max_sentences]
    answer = " ".join(sentence for _, _, sentence in sorted(best, key=lambda item: item[1]))
    return _strip_wrapping_code_fences(answer or contexts[0][:500]).replace("```", "").strip()


def _is_low_value_code_evidence(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.match(r"^import\b", stripped)
        or re.match(r"^export\s+{", stripped)
        or stripped.startswith("Focused excerpt from ")
        or stripped.startswith("} from ")
        or " from '@nestjs/" in stripped
        or " from 'react" in stripped
        or re.match(r"^from\s+['\"A-Za-z0-9_./@-]+", stripped)
        or stripped in {"{", "}", "});", "};", ");"}
    )


def synthesize_source_grounded_answer(
    query: str,
    contexts: Sequence[str],
    sources: Sequence[dict[str, Any]],
    max_sources: int = 5,
) -> str | None:
    analysis = analyze_query(query)
    query_terms = _significant_terms(analysis.rewritten_query)
    evidence_rows: list[tuple[float, int, str]] = []
    seen_sources: set[str] = set()
    has_code_sources = any(_is_code_evidence_source(str(item.get("source_doc", ""))) for item in sources)
    prefer_project_docs = _query_prefers_project_documentation(query, analysis)

    for index, context in enumerate(contexts):
        source = sources[index].get("source_doc", f"document_{index + 1}") if index < len(sources) else f"document_{index + 1}"
        source = str(source)
        if source in seen_sources:
            continue
        if _is_readme_source(source) and has_code_sources and not prefer_project_docs:
            continue
        if _is_maintenance_source(source, analysis):
            continue
        if _is_manifest_source(source, analysis):
            continue
        seen_sources.add(source)

        lines = [
            line.strip()
            for line in _context_for_synthesis(context).splitlines()
            if line.strip() and not _is_low_value_code_evidence(line)
        ]
        ranked: list[tuple[float, int, str]] = []
        for line_index, line in enumerate(lines):
            normalized_line = _normalize_for_match(line)
            terms = _significant_terms(normalized_line)
            overlap = len(query_terms & terms)
            specific_overlap = len((query_terms - LOW_VALUE_CODE_QUERY_TERMS) & terms)
            path_bonus = _source_scope_score(source, analysis)
            if specific_overlap <= 0 and path_bonus < 1.0:
                continue
            code_bonus = 0.3 if re.search(r"\b(async|await|return|throw|if|const|let|class|function|private|public|static)\b", line) else 0.0
            ranked.append((specific_overlap + (0.25 * overlap) + path_bonus + code_bonus, line_index, line))

        ranked = [item for item in ranked if item[0] > 0]
        if not ranked:
            continue
        best_lines = [line for _, _, line in sorted(sorted(ranked, key=lambda item: (-item[0], item[1]))[:5], key=lambda item: item[1])]
        evidence = " ".join(best_lines)
        if len(evidence) > 320:
            evidence = f"{evidence[:317].rstrip()}..."
        row_score = max(score for score, _, _ in ranked) + 0.05 * float(sources[index].get("score") or 0) if index < len(sources) else max(score for score, _, _ in ranked)
        evidence_rows.append((row_score, index, f"- `{source}`: {evidence}"))

    if not evidence_rows:
        return None
    bullets = [
        bullet
        for _, _, bullet in sorted(evidence_rows, key=lambda item: (-item[0], item[1]))[:max_sources]
    ]
    return "Evidence from retrieved source files:\n" + "\n".join(bullets)


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

    def _load_existing_index(self):
        """Try to load the persisted ChromaDB index without rebuilding.

        Returns (VectorStoreIndex, nodes) if chroma_db exists and has documents,
        otherwise (None, None). Only uses local embedding model — no network calls.
        """
        chroma_dir = PROJECT_ROOT / "chroma_db"
        if not chroma_dir.exists():
            return None, None

        try:
            import os as _os

            _os.environ.setdefault("HF_HUB_OFFLINE", "1")

            import chromadb

            from llama_index.core import VectorStoreIndex
            from llama_index.core.schema import TextNode
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.vector_stores.chroma import ChromaVectorStore

            chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = chroma_client.get_collection("advanced_rag")
            if collection.count() == 0:
                return None, None
            vector_store = ChromaVectorStore(chroma_collection=collection)
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=embed_model
            )
            all_data = collection.get(include=["documents", "metadatas"])
            nodes = [
                TextNode(
                    text=doc,
                    metadata=meta or {},
                )
                for doc, meta in zip(
                    all_data.get("documents", []) or [],
                    all_data.get("metadatas", []) or [],
                )
                if doc
            ]
            return index, nodes
        except Exception:
            logger.debug("Failed to load existing ChromaDB index", exc_info=True)
            return None, None

    def _ensure_ready(self) -> None:
        if self.retriever is not None:
            return

        if not self.allow_index_build:
            self.nodes = load_local_context_nodes()
            index, chroma_nodes = self._load_existing_index()
            if index is not None and chroma_nodes:
                self.index = index
                self.nodes = chroma_nodes
                self.retriever = self._build_retriever()
                return
            self.retriever = SimpleLocalRetriever(self.nodes, top_k=self.top_k)
            return

        if not _env_enabled_by_default("ALLOW_INDEX_BUILD"):
            raise RuntimeError(
                "Index build requested but ALLOW_INDEX_BUILD=0. "
                "Unset it or set ALLOW_INDEX_BUILD=1 to enable index building."
            )

        from ingestion import build_index

        self.index, self.nodes = build_index()
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        if self.index is None:
            logger.info("No vector index found, using SimpleLocalRetriever (lexical-only)")
            return SimpleLocalRetriever(self.nodes or [], top_k=self.top_k)

        from retrieval import HybridRetriever

        logger.info("Vector index available, using HybridRetriever")
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

        logger.info("answer_query: strategy=%s, query=%s", strategy, query[:80])
        self._ensure_ready()
        results, metadata = self.retriever.ablation_retrieve(query, strategy)
        results, deepening_trace = _deepen_code_evidence_results(query, results, self.nodes or [], self.top_k)
        if deepening_trace["enabled"]:
            metadata = {**metadata, "context_deepening": deepening_trace}
        results, project_trace = _balance_project_evidence_results(query, results, self.nodes or [], self.top_k)
        if project_trace["enabled"]:
            metadata = {**metadata, "project_evidence": project_trace}
        results, stack_trace = _balance_stack_evidence_results(query, results, self.nodes or [], self.top_k)
        if stack_trace["enabled"]:
            metadata = {**metadata, "stack_evidence": stack_trace}
        contexts = [_node_text(result) for result in results]
        logger.info("answer_query: retrieved %d contexts, %d results", len(contexts), len(results))
        sources = self._source_rows(results)
        stack_answer = _format_stack_answer(contexts, sources, query) if "stack" in analyze_query(query).intents else None
        answer = stack_answer or (synthesize_extractive_answer(query, contexts) if contexts else EMPTY_LOCAL_CONTEXT_ANSWER)

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
        results, deepening_trace = _deepen_code_evidence_results(retrieval_query, results, self.nodes or [], self.top_k)
        if deepening_trace["enabled"]:
            metadata = {**metadata, "context_deepening": deepening_trace}
        results, project_trace = _balance_project_evidence_results(retrieval_query, results, self.nodes or [], self.top_k)
        if project_trace["enabled"]:
            metadata = {**metadata, "project_evidence": project_trace}
        results, stack_trace = _balance_stack_evidence_results(retrieval_query, results, self.nodes or [], self.top_k)
        if stack_trace["enabled"]:
            metadata = {**metadata, "stack_evidence": stack_trace}
        contexts = [_node_text(result) for result in results]
        sources = self._source_rows(results)
        citations = _make_citations(results)
        fine_tune_metadata = None
        if intent == "fine_tune":
            fine_tune_metadata = _extract_fine_tune_info(contexts, sources)
        synthesis_result = synthesize_chat_answer(message, contexts, intent, sources, fine_tune_metadata=fine_tune_metadata)
        trace = self._normalize_trace(metadata, results)
        trace.update(
            {
                "history_items": len(history),
                "intent": intent,
                "intents": analysis.intents,
                "evidence_score": round(_evidence_score(message, contexts), 3),
                "synthesis": synthesis_result.to_trace(),
            }
        )

        return {
            "answer": synthesis_result.answer,
            "citations": citations,
            "contexts": contexts,
            "sources": sources,
            "trace": trace,
            "intent": intent,
            "confidence": _confidence(message, contexts, sources),
        }


def answer_query(query: str, strategy: str = "hybrid_rerank", allow_index_build: bool = True) -> dict[str, Any]:
    return LocalRAGPipeline(allow_index_build=allow_index_build).answer_query(query, strategy=strategy)


def chat_query(
    message: str,
    history: Sequence[dict[str, Any]] | None = None,
    strategy: str = "hybrid_rerank",
    allow_index_build: bool = True,
) -> dict[str, Any]:
    return LocalRAGPipeline(allow_index_build=allow_index_build).chat_query(
        message,
        history=history,
        strategy=strategy,
    )
