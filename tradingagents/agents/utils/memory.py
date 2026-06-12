"""Financial situation memory — hybrid BM25 + dense retrieval (Tier B8).

Default path remains BM25Okapi for full back-compat with existing
backtest replays. When ``config["hybrid_rag"]=True`` the class also
builds a dense FAISS index over sentence-transformers embeddings and
fuses BM25 + dense ranks via reciprocal-rank-fusion (Akarsu et al.
arXiv 2604.01733: hybrid + cross-encoder rerank wins R@5 = 0.816 on
financial RAG benchmarks). Dense path lazy-imports
sentence-transformers + faiss so it's optional.
"""

from rank_bm25 import BM25Okapi
from typing import List, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class _DenseRetriever:
    """Optional dense-vector retriever; lazy-imports sentence-transformers + faiss.

    Falls back silently if either dependency is missing — the parent
    ``FinancialSituationMemory`` then operates BM25-only as before.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._index = None
        self._enabled = False
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            import faiss  # noqa: F401
            self._enabled = True
        except Exception as exc:  # noqa: BLE001
            logger.info(f"hybrid_rag dense path disabled ({exc.__class__.__name__})")

    def _ensure_model(self):
        if self._model is None and self._enabled:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def build(self, docs: List[str]) -> None:
        if not self._enabled or not docs:
            self._index = None
            return
        try:
            import faiss
            import numpy as np
            self._ensure_model()
            embs = self._model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embs.astype("float32"))
            self._index = index
            self._dim = dim
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"dense index build failed: {exc}")
            self._index = None

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if not self._enabled or self._index is None:
            return []
        try:
            import numpy as np
            self._ensure_model()
            q = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores, idx = self._index.search(q.astype("float32"), k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"dense search failed: {exc}")
            return []


def _reciprocal_rank_fusion(
    rankings: List[List[int]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Standard RRF over multiple ranked lists. Higher score = better."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for r, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class FinancialSituationMemory:
    """Memory system — BM25-only by default; optional hybrid dense fusion.

    ``config["hybrid_rag"] = True`` activates the dense path. Existing
    callers see no behavior change.
    """

    def __init__(self, name: str, config: Optional[dict] = None):
        self.name = name
        self.documents: List[str] = []
        self.recommendations: List[str] = []
        self.bm25 = None
        cfg = config or {}
        self._hybrid = bool(cfg.get("hybrid_rag", False))
        self._dense: Optional[_DenseRetriever] = (
            _DenseRetriever() if self._hybrid else None
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing.

        Simple whitespace + punctuation tokenization with lowercasing.
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _rebuild_index(self):
        """Rebuild the BM25 (and optional dense) index after adding documents."""
        if self.documents:
            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None
        if self._dense is not None:
            self._dense.build(self.documents)

    def add_situations(self, situations_and_advice: List[Tuple[str, str]]):
        """Add financial situations and their corresponding advice.

        Args:
            situations_and_advice: List of tuples (situation, recommendation)
        """
        for situation, recommendation in situations_and_advice:
            self.documents.append(situation)
            self.recommendations.append(recommendation)

        # Rebuild BM25 index with new documents
        self._rebuild_index()

    def get_memories(self, current_situation: str, n_matches: int = 1) -> List[dict]:
        """Find matching recommendations using BM25 similarity.

        Args:
            current_situation: The current financial situation to match against
            n_matches: Number of top matches to return

        Returns:
            List of dicts with matched_situation, recommendation, and similarity_score
        """
        if not self.documents or self.bm25 is None:
            return []

        # BM25 ranking
        query_tokens = self._tokenize(current_situation)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranking = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )

        # Hybrid fusion path
        if self._dense is not None and self._dense._enabled and self._dense._index is not None:
            dense_hits = self._dense.search(current_situation, k=min(20, len(self.documents)))
            dense_ranking = [doc_id for doc_id, _ in dense_hits]
            fused = _reciprocal_rank_fusion(
                [bm25_ranking[: min(20, len(self.documents))], dense_ranking]
            )
            top = [doc_id for doc_id, _ in fused[:n_matches]]
            results = []
            max_fused = fused[0][1] if fused else 1.0
            for doc_id, score in fused[:n_matches]:
                results.append({
                    "matched_situation": self.documents[doc_id],
                    "recommendation": self.recommendations[doc_id],
                    "similarity_score": float(score / max_fused) if max_fused else 0.0,
                })
            logger.info(
                f"Hybrid retrieval: K_bm25={len(bm25_ranking)}, "
                f"K_dense={len(dense_ranking)}, K_final={len(results)}"
            )
            return results

        # BM25-only fallback (default)
        top_indices = bm25_ranking[:n_matches]
        max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
        return [
            {
                "matched_situation": self.documents[idx],
                "recommendation": self.recommendations[idx],
                "similarity_score": float(bm25_scores[idx] / max_score) if max_score else 0.0,
            }
            for idx in top_indices
        ]

    def clear(self):
        """Clear all stored memories."""
        self.documents = []
        self.recommendations = []
        self.bm25 = None


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory("test_memory")

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
