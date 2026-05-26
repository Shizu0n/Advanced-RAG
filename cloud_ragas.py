"""Gemini free-tier RAGAS helpers with explicit no-default backends."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "eval" / "cache"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
QUOTA_STATUS_CODES = {403, 429, 500, 502, 503, 504}
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
SUPPORTED_CLOUD_PROVIDERS = ("gemini", "github", "groq")
DEFAULT_PROVIDER_ORDER = ("gemini", "github", "groq")


class CloudProviderUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CloudRagasConfig:
    providers: tuple["CloudProvider", ...]
    cache_dir: Path = DEFAULT_CACHE_DIR
    max_calls: int = 120


@dataclass
class CloudCallBudget:
    max_calls: int = 120
    calls_made: int = 0

    def consume(self) -> None:
        if self.calls_made >= self.max_calls:
            raise RuntimeError(f"MAX_CLOUD_CALLS exceeded ({self.max_calls}); cached calls still work.")
        self.calls_made += 1


@dataclass(frozen=True)
class CloudProvider:
    name: str
    model: str
    api_key: str
    account_id: str | None = None


def _provider_order_from_env() -> tuple[str, ...]:
    raw_order = os.getenv("CLOUD_PROVIDER_ORDER")
    if not raw_order:
        return DEFAULT_PROVIDER_ORDER
    order = tuple(name.strip().lower() for name in raw_order.split(",") if name.strip())
    unsupported = [name for name in order if name not in SUPPORTED_CLOUD_PROVIDERS]
    if unsupported:
        raise RuntimeError(f"Unsupported CLOUD_PROVIDER_ORDER provider: {unsupported[0]}")
    return order or DEFAULT_PROVIDER_ORDER


def providers_from_env() -> list[CloudProvider]:
    configured: dict[str, CloudProvider] = {}
    if os.getenv("GEMINI_API_KEY"):
        configured["gemini"] = CloudProvider("gemini", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL), os.environ["GEMINI_API_KEY"])
    if os.getenv("GITHUB_MODELS_TOKEN") and os.getenv("GITHUB_MODELS_MODEL"):
        model = os.environ["GITHUB_MODELS_MODEL"]
        if _is_allowed_fallback_model(model):
            configured["github"] = CloudProvider("github", model, os.environ["GITHUB_MODELS_TOKEN"])
    if os.getenv("GROQ_API_KEY"):
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if _is_allowed_fallback_model(model):
            configured["groq"] = CloudProvider("groq", model, os.environ["GROQ_API_KEY"])
    return [configured[name] for name in _provider_order_from_env() if name in configured]




def _is_allowed_fallback_model(model: str) -> bool:
    normalized = model.lower()
    blocked_markers = ("gemini", "google/", "gemma")
    return not any(marker in normalized for marker in blocked_markers)


def config_from_env() -> CloudRagasConfig:
    if os.getenv("ALLOW_CLOUD_FREE_TIER") != "1":
        raise RuntimeError("Set ALLOW_CLOUD_FREE_TIER=1 to permit free-tier cloud evaluation.")
    providers = tuple(providers_from_env())
    if not providers:
        raise RuntimeError("Set at least one supported free-tier cloud provider key before cloud RAGAS evaluation.")
    max_calls = int(os.getenv("MAX_CLOUD_CALLS", "120"))
    return CloudRagasConfig(providers=providers, max_calls=max_calls)


def cloud_ragas_enabled() -> bool:
    return os.getenv("USE_CLOUD_FREE_TIER_RAGAS") == "1"


def client_from_config(config: CloudRagasConfig) -> "FreeTierCloudClient":
    return FreeTierCloudClient(cache_dir=config.cache_dir, budget=CloudCallBudget(max_calls=config.max_calls), providers=config.providers)


class FreeTierCloudClient:
    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        post: Callable[..., Any] = requests.post,
        max_calls: int = 120,
        budget: CloudCallBudget | None = None,
        providers: Sequence[CloudProvider] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.post = post
        self.budget = budget or CloudCallBudget(max_calls=max_calls)
        self.providers = tuple(providers or providers_from_env())
        if not self.providers:
            raise RuntimeError("Set at least one supported free-tier cloud provider key.")

    @property
    def calls_made(self) -> int:
        return self.budget.calls_made

    def generate_text(self, prompt: Any, n: int = 1, temperature: float | None = None) -> str:
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        logger.info("generate_text: %d chars, providers=%s", len(text), [p.name for p in self.providers])
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "candidateCount": n,
                "temperature": 0.0 if temperature is None else temperature,
            },
        }
        data = self._request_with_provider_fallback("generate_text", payload, self.providers)
        logger.info("generate_text: success, response=%d chars", len(str(data)))
        return self._text_from_response(data)

    async def agenerate_text(self, prompt: Any, n: int = 1, temperature: float | None = None) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_text, prompt, n, temperature)

    def generate_json(self, prompt: str) -> dict[str, Any]:
        raw = self.generate_text(prompt, temperature=0.0)
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)

    def embed_text(self, text: str) -> list[float]:
        payload = {
            "model": f"models/{GEMINI_EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text}]},
        }
        data = self._request_with_model_fallback("embedContent", payload, (GEMINI_EMBEDDING_MODEL,))
        values = data.get("embedding", {}).get("values", [])
        return [float(value) for value in values]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    async def aembed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, list(texts))

    def _request_with_provider_fallback(
        self,
        method: str,
        payload: dict[str, Any],
        providers: Sequence[CloudProvider],
    ) -> dict[str, Any]:
        errors: list[str] = []
        for provider in providers:
            cache_payload = {"provider": provider.name, "model": provider.model, "payload": payload}
            cache_key = self._cache_key(method, provider.model, cache_payload)
            cached = self._read_cache(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s/%s", provider.name, provider.model)
                return cached

            logger.info("Trying provider %s/%s (budget: %d/%d)", provider.name, provider.model, self.budget.calls_made, self.budget.max_calls)
            self.budget.consume()
            url, request_payload, headers = self._provider_request(provider, payload)
            response = self.post(url, json=request_payload, headers=headers, timeout=30)
            if getattr(response, "status_code", 200) in QUOTA_STATUS_CODES:
                logger.warning("Provider %s returned HTTP %s, skipping", provider.name, response.status_code)
                errors.append(f"{provider.name}/{provider.model}: HTTP {response.status_code}")
                continue
            response.raise_for_status()
            data = response.json()
            self._write_cache(cache_key, data)
            logger.info("Provider %s succeeded", provider.name)
            return data
        raise CloudProviderUnavailable("; ".join(errors) or "cloud providers unavailable")

    def _provider_request(self, provider: CloudProvider, payload: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, str]]:
        prompt = payload["contents"][0]["parts"][0]["text"]
        temperature = payload.get("generationConfig", {}).get("temperature", 0.0)
        if provider.name == "gemini":
            return self._url(provider.model, "generateContent"), payload, self._gemini_headers(provider.api_key)

        messages = [{"role": "user", "content": prompt}]
        if provider.name == "groq":
            return (
                "https://api.groq.com/openai/v1/chat/completions",
                {"model": provider.model, "messages": messages, "temperature": temperature},
                {"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"},
            )
        if provider.name == "github":
            return (
                "https://models.github.ai/inference/chat/completions",
                {"model": provider.model, "messages": messages, "temperature": temperature},
                {
                    "Authorization": f"Bearer {provider.api_key}",
                    "Accept": "application/vnd.github+json",
                    "Content-Type": "application/json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        raise RuntimeError(f"Unsupported cloud provider: {provider.name}")

    def _text_from_response(self, data: dict[str, Any]) -> str:
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        if "result" in data and isinstance(data["result"], dict):
            return data["result"].get("response") or data["result"].get("text") or ""
        raise ValueError("Cloud provider response did not contain generated text.")

    def _gemini_provider(self) -> CloudProvider:
        for provider in self.providers:
            if provider.name == "gemini":
                return provider
        raise CloudProviderUnavailable("Gemini embeddings require GEMINI_API_KEY.")

    def _request_with_model_fallback(
        self,
        method: str,
        payload: dict[str, Any],
        models: Sequence[str],
    ) -> dict[str, Any]:
        errors: list[str] = []
        provider = self._gemini_provider()
        for model in models:
            cache_key = self._cache_key(method, model, payload)
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

            url = self._url(model, method)
            self.budget.consume()
            response = self.post(url, json=payload, headers=self._gemini_headers(provider.api_key), timeout=30)
            if getattr(response, "status_code", 200) in QUOTA_STATUS_CODES:
                errors.append(f"{model}: HTTP {response.status_code}")
                continue
            response.raise_for_status()
            data = response.json()
            self._write_cache(cache_key, data)
            return data
        raise CloudProviderUnavailable("; ".join(errors) or "Gemini embedding models unavailable")

    def _url(self, model: str, method: str) -> str:
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}"

    def _gemini_headers(self, api_key: str) -> dict[str, str]:
        return {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def _cache_key(self, method: str, model: str, payload: dict[str, Any]) -> str:
        raw = json.dumps({"method": method, "model": model, "payload": payload}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> dict[str, Any] | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_cache(self, cache_key: str, data: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(cache_key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def build_cloud_backends(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_calls: int = 120,
    cloud_client: FreeTierCloudClient | None = None,
) -> tuple[Any, Any]:
    client = cloud_client or FreeTierCloudClient(cache_dir=cache_dir, max_calls=max_calls)

    try:
        from ragas.embeddings import BaseRagasEmbeddings
        from ragas.llms import BaseRagasLLM
        from langchain_core.outputs import Generation, LLMResult
    except ImportError:
        return PlainCloudLLM(client), PlainLocalEmbeddings(client)

    class CloudRagasLLM(BaseRagasLLM):
        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = None,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            text = client.generate_text(prompt, n=n, temperature=temperature)
            return LLMResult(generations=[[Generation(text=text)]])

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = None,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            text = await client.agenerate_text(prompt, n=n, temperature=temperature)
            return LLMResult(generations=[[Generation(text=text)]])

    class LocalRagasEmbeddings(BaseRagasEmbeddings):
        def embed_query(self, text: str) -> list[float]:
            return _local_embedding(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [_local_embedding(text) for text in texts]

        async def aembed_query(self, text: str) -> list[float]:
            return _local_embedding(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return [_local_embedding(text) for text in texts]

    return CloudRagasLLM(), LocalRagasEmbeddings()


class PlainCloudLLM:
    def __init__(self, client: FreeTierCloudClient) -> None:
        self.client = client


class PlainLocalEmbeddings:
    def __init__(self, client: FreeTierCloudClient) -> None:
        self.client = client


def _local_embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def _import_ragas_parts() -> dict[str, Any]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    return {
        "Dataset": Dataset,
        "evaluate": evaluate,
        "metrics": [faithfulness, answer_relevancy, context_recall, context_precision],
    }


def run_ragas(
    rows: Sequence[dict[str, Any]],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_calls: int = 120,
    cloud_client: FreeTierCloudClient | None = None,
) -> dict[str, float]:
    parts = _import_ragas_parts()
    llm, embeddings = build_cloud_backends(
        cache_dir=cache_dir,
        max_calls=max_calls,
        cloud_client=cloud_client,
    )
    records = [_ragas_record(row) for row in rows]
    dataset = parts["Dataset"].from_list(records)
    result = parts["evaluate"](
        dataset,
        metrics=parts["metrics"],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=True,
        show_progress=False,
    )
    return _result_to_scores(result)


def _ragas_record(row: dict[str, Any]) -> dict[str, Any]:
    contexts = row["contexts"]
    if isinstance(contexts, str):
        contexts = json.loads(contexts)
    return {
        "user_input": row["question"],
        "response": row["answer"],
        "retrieved_contexts": list(contexts),
        "reference": row["ground_truth"],
    }


def _result_to_scores(result: Any) -> dict[str, float]:
    if hasattr(result, "to_pandas"):
        data = result.to_pandas().to_dict()
    elif isinstance(result, dict):
        data = result
    else:
        data = dict(result)

    scores: dict[str, float] = {}
    for metric in METRIC_NAMES:
        value = data.get(metric)
        if isinstance(value, dict):
            values = [float(item) for item in value.values()]
            scores[metric] = sum(values) / max(len(values), 1)
        elif isinstance(value, list):
            scores[metric] = sum(float(item) for item in value) / max(len(value), 1)
        elif value is not None:
            scores[metric] = float(value)
    return scores
