import unittest
from unittest.mock import patch

from llama_index.core.schema import NodeWithScore, TextNode

from retrieval import HybridRetriever


class StaticRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query):
        return list(self.results)


class FakeCrossEncoder:
    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text

    def predict(self, pairs):
        return [self.scores_by_text[text] for _, text in pairs]


def result(node_id, text, score=1.0):
    return NodeWithScore(node=TextNode(id_=node_id, text=text), score=score)


class HybridRetrieverTests(unittest.TestCase):
    def test_rrf_prefers_documents_that_appear_in_multiple_rankings(self):
        vector = [result("a", "semantic"), result("b", "shared")]
        bm25 = [result("b", "shared"), result("c", "keyword")]

        retriever = HybridRetriever(
            index=None,
            nodes=[],
            top_k=2,
            vector_retriever=StaticRetriever(vector),
            bm25_retriever=StaticRetriever(bm25),
            cross_encoder=FakeCrossEncoder({"semantic": 0.1, "shared": 0.2, "keyword": 0.3}),
        )

        fused = retriever._reciprocal_rank_fusion([vector, bm25])

        self.assertEqual(fused[0].node.node_id, "b")
        self.assertGreater(fused[0].score, fused[1].score)

    def test_hybrid_rerank_uses_cross_encoder_scores_and_top_k(self):
        vector = [result("a", "semantic"), result("b", "shared")]
        bm25 = [result("b", "shared"), result("c", "keyword")]

        retriever = HybridRetriever(
            index=None,
            nodes=[],
            top_k=2,
            vector_retriever=StaticRetriever(vector),
            bm25_retriever=StaticRetriever(bm25),
            cross_encoder=FakeCrossEncoder({"semantic": 0.2, "shared": 0.1, "keyword": 0.9}),
        )

        results, metadata = retriever.ablation_retrieve("query", "hybrid_rerank")

        self.assertEqual([item.node.node_id for item in results], ["c", "a"])
        self.assertEqual(metadata["strategy"], "hybrid_rerank")
        self.assertEqual(metadata["reranker"], "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertEqual(metadata["vector_scores"][0], {"source": "a", "score": 1.0})
        self.assertEqual(metadata["bm25_scores"][0], {"source": "b", "score": 1.0})
        self.assertTrue(metadata["rrf_scores"])
        self.assertEqual(metadata["reranker_scores"][0], {"source": "c", "score": 0.9})

    def test_ablation_can_return_single_strategy_results(self):
        vector = [result("a", "semantic")]
        bm25 = [result("b", "keyword")]

        retriever = HybridRetriever(
            index=None,
            nodes=[],
            top_k=5,
            vector_retriever=StaticRetriever(vector),
            bm25_retriever=StaticRetriever(bm25),
            cross_encoder=FakeCrossEncoder({"semantic": 0.2, "keyword": 0.8}),
        )

        semantic_results, semantic_metadata = retriever.ablation_retrieve("query", "semantic_only")
        bm25_results, bm25_metadata = retriever.ablation_retrieve("query", "bm25_only")

        self.assertEqual([item.node.node_id for item in semantic_results], ["a"])
        self.assertEqual([item.node.node_id for item in bm25_results], ["b"])
        self.assertFalse(semantic_metadata["used_bm25"])
        self.assertFalse(bm25_metadata["used_vector"])
        self.assertEqual(semantic_metadata["vector_scores"], [{"source": "a", "score": 1.0}])
        self.assertEqual(semantic_metadata["bm25_scores"], [])
        self.assertEqual(bm25_metadata["bm25_scores"], [{"source": "b", "score": 1.0}])
        self.assertEqual(bm25_metadata["vector_scores"], [])

    def test_default_reranker_requires_explicit_model_download_opt_in(self):
        vector = [result("a", "semantic")]
        bm25 = [result("b", "keyword")]
        retriever = HybridRetriever(
            index=None,
            nodes=[],
            top_k=2,
            vector_retriever=StaticRetriever(vector),
            bm25_retriever=StaticRetriever(bm25),
        )

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ALLOW_MODEL_DOWNLOADS"):
                retriever.ablation_retrieve("query", "hybrid_rerank")


if __name__ == "__main__":
    unittest.main()
