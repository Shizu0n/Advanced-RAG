"""HybridRetriever: BM25 + vector + RRF + reranking."""

from __future__ import annotations

import logging
import re
import os

logger = logging.getLogger(__name__)
from collections import defaultdict
from typing import Iterable, Sequence

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode, NodeWithScore
from rank_bm25 import BM25Okapi


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9]+", text):
        parts = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", raw).split()
        tokens.extend(part.lower() for part in parts if part)
    return tokens


def _node_text(item: BaseNode | NodeWithScore) -> str:
    node = item.node if isinstance(item, NodeWithScore) else item
    return node.get_content() if hasattr(node, "get_content") else node.text


def _node_id(item: BaseNode | NodeWithScore) -> str:
    node = item.node if isinstance(item, NodeWithScore) else item
    return node.node_id


def _as_node_with_score(item: BaseNode | NodeWithScore, score: float) -> NodeWithScore:
    node = item.node if isinstance(item, NodeWithScore) else item
    return NodeWithScore(node=node, score=float(score))


def _score_rows(results: Sequence[NodeWithScore]) -> list[dict[str, float | str | None]]:
    return [
        {"source": _node_id(result), "score": float(result.score) if result.score is not None else None}
        for result in results
    ]


class LocalLexicalPositionReranker:
    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, text in pairs:
            query_terms = set(_tokenize(query))
            text_tokens = _tokenize(text)
            if not query_terms or not text_tokens:
                scores.append(0.0)
                continue
            text_terms = set(text_tokens)
            coverage = len(query_terms & text_terms) / len(query_terms)
            density = len(query_terms & text_terms) / len(text_terms)
            first_positions = [text_tokens.index(term) for term in query_terms if term in text_tokens]
            position = 1.0 / (1.0 + min(first_positions)) if first_positions else 0.0
            scores.append(coverage + density + position)
        return scores


class BM25Retriever:
    """Small BM25 adapter that returns LlamaIndex NodeWithScore objects."""

    def __init__(self, nodes: Sequence[BaseNode], top_k: int = 10) -> None:
        self.nodes = list(nodes)
        self.top_k = top_k
        self._tokenized_nodes = [_tokenize(_node_text(node)) for node in self.nodes]
        self._bm25 = BM25Okapi(self._tokenized_nodes) if self.nodes else None

    def retrieve(self, query: str) -> list[NodeWithScore]:
        if not self._bm25:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results = [
            NodeWithScore(node=self.nodes[index], score=float(score))
            for index, score in ranked[: self.top_k]
            if score > 0
        ]
        if results:
            return results
        query_terms = set(query_tokens)
        lexical = [
            (index, len(query_terms & set(tokens)) / max(len(query_terms), 1))
            for index, tokens in enumerate(self._tokenized_nodes)
        ]
        return [
            NodeWithScore(node=self.nodes[index], score=float(score))
            for index, score in sorted(lexical, key=lambda item: item[1], reverse=True)[: self.top_k]
            if score > 0
        ]


class HybridRetriever:
    def __init__(
        self,
        index,
        nodes: Sequence[BaseNode],
        top_k: int = 5,
        vector_retriever=None,
        bm25_retriever=None,
        cross_encoder=None,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
    ) -> None:
        self.index = index
        self.nodes = list(nodes)
        self.top_k = top_k
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.cross_encoder_model = cross_encoder_model
        self.vector_retriever = vector_retriever or VectorIndexRetriever(
            index=index,
            similarity_top_k=vector_top_k,
        )
        self.bm25_retriever = bm25_retriever or BM25Retriever(self.nodes, top_k=bm25_top_k)
        self.cross_encoder = cross_encoder

    def _get_cross_encoder(self):
        if self.cross_encoder is not None:
            return self.cross_encoder

        if os.getenv("ALLOW_MODEL_DOWNLOADS") == "1":
            from sentence_transformers import CrossEncoder

            self.cross_encoder = CrossEncoder(self.cross_encoder_model)
            return self.cross_encoder

        return LocalLexicalPositionReranker()

    def retrieve(self, query: str) -> list[NodeWithScore]:
        results, _ = self.ablation_retrieve(query, "hybrid_rerank")
        return results

    def _reciprocal_rank_fusion(
        self,
        results_lists: Iterable[Sequence[NodeWithScore]],
        k: int = 60,
        limit: int = 20,
    ) -> list[NodeWithScore]:
        scores: defaultdict[str, float] = defaultdict(float)
        best_node: dict[str, NodeWithScore] = {}

        for results in results_lists:
            for rank, result in enumerate(results, start=1):
                node_key = _node_id(result)
                scores[node_key] += 1.0 / (k + rank)
                best_node.setdefault(node_key, result)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [_as_node_with_score(best_node[node_key], score) for node_key, score in ranked[:limit]]

    def _rerank(self, query: str, nodes: Sequence[NodeWithScore]) -> list[NodeWithScore]:
        if not nodes:
            return []

        pairs = [(query, _node_text(node)) for node in nodes]
        scores = self._get_cross_encoder().predict(pairs)
        reranked = [
            _as_node_with_score(node, float(score))
            for node, score in zip(nodes, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: item.score or 0.0, reverse=True)[: self.top_k]

    def ablation_retrieve(self, query: str, strategy: str) -> tuple[list[NodeWithScore], dict]:
        logger.info("ablation_retrieve: strategy=%s, query=%s", strategy, query[:80])
        vector_results: list[NodeWithScore] = []
        bm25_results: list[NodeWithScore] = []
        fused_results: list[NodeWithScore] = []

        if strategy in {"semantic_only", "hybrid_no_rerank", "hybrid_rerank"}:
            vector_results = self.vector_retriever.retrieve(query)[: self.vector_top_k]
            logger.debug("vector retrieval: %d results", len(vector_results))

        if strategy in {"bm25_only", "hybrid_no_rerank", "hybrid_rerank"}:
            bm25_results = self.bm25_retriever.retrieve(query)[: self.bm25_top_k]
            logger.debug("bm25 retrieval: %d results", len(bm25_results))

        if strategy == "semantic_only":
            results = vector_results[: self.top_k]
        elif strategy == "bm25_only":
            results = bm25_results[: self.top_k]
        elif strategy == "hybrid_no_rerank":
            fused_results = self._reciprocal_rank_fusion([vector_results, bm25_results])
            results = fused_results[: self.top_k]
        elif strategy == "hybrid_rerank":
            fused_results = self._reciprocal_rank_fusion([vector_results, bm25_results])
            results = self._rerank(query, fused_results)
        else:
            raise ValueError(
                "strategy must be one of: semantic_only, bm25_only, "
                "hybrid_no_rerank, hybrid_rerank"
            )

        metadata = {
            "strategy": strategy,
            "top_k": self.top_k,
            "vector_top_k": self.vector_top_k,
            "bm25_top_k": self.bm25_top_k,
            "vector_count": len(vector_results),
            "bm25_count": len(bm25_results),
            "fused_count": len(fused_results),
            "used_vector": bool(vector_results),
            "used_bm25": bool(bm25_results),
            "used_rerank": strategy == "hybrid_rerank",
            "reranker": (
                self.cross_encoder_model
                if strategy == "hybrid_rerank" and (self.cross_encoder is not None or os.getenv("ALLOW_MODEL_DOWNLOADS") == "1")
                else "local_lexical_position" if strategy == "hybrid_rerank" else None
            ),
            "vector_scores": _score_rows(vector_results),
            "bm25_scores": _score_rows(bm25_results),
            "rrf_scores": _score_rows(fused_results),
            "reranker_scores": _score_rows(results) if strategy == "hybrid_rerank" else [],
        }
        return results, metadata
