"""
embedder.py — Multilingual embedding management.

Wraps sentence-transformers to encode:
  • seed concepts  → fixed matrix  E_C  (M × d)
  • document segments → matrix    E_P  (N_segs × d)
  • phrases (Tier 2) → on-demand

Similarity is computed with NumPy. Models load only when encoding is requested.
"""

from __future__ import annotations

import logging
import pickle
from typing import List, Optional, Tuple

import numpy as np

from .seed_concepts import get_concept_ids

logger = logging.getLogger(__name__)



# LAZY IMPORT HELPERS


def _load_sbert(model_name: str, device: str = "cpu", cache_dir: str = "models/"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install sentence-transformers")
    return SentenceTransformer(model_name, device=device, cache_folder=cache_dir)


def _embedding_dimension(model) -> int:
    """Return embedding dimension across sentence-transformers API versions."""
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())



# COSINE UTILITIES


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (in-place safe)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between every row in A and every row in B.

    Parameters
    ----------
    A : (n, d)
    B : (m, d)

    Returns
    -------
    sim : (n, m)  where sim[i, j] = cos(A[i], B[j])
    """
    A_n = l2_normalize(A)
    B_n = l2_normalize(B)
    return A_n @ B_n.T  # (n, d) @ (d, m) = (n, m)


def top_r_mean_similarity(
    query_vec: np.ndarray,   # (d,)
    seed_matrix: np.ndarray, # (M, d)
    r: int = 5,
) -> float:
    """
    R_j = (1/r) Σ_{m ∈ N_r(p_j)} cos(e(p_j), e(c_m))

    Average over the r closest seed concepts.
    """
    sims = cosine_similarity_matrix(
        query_vec.reshape(1, -1), seed_matrix
    )[0]  # (M,)
    top_r_indices = np.argsort(sims)[::-1][:r]
    return float(np.mean(sims[top_r_indices]))


def max_seed_similarity(
    query_vec: np.ndarray,    # (d,)
    seed_matrix: np.ndarray,  # (M, d)
) -> float:
    """R_j = max_m cos(e(p_j), e(c_m))"""
    sims = cosine_similarity_matrix(
        query_vec.reshape(1, -1), seed_matrix
    )[0]
    return float(np.max(sims))



# MAIN EMBEDDER CLASS


class FinTechEmbedder:
    """
    Central embedding engine for the FTA pipeline.

    Usage
    -----
    embedder = FinTechEmbedder(model_name="paraphrase-multilingual-mpnet-base-v2")
    embedder.encode_seeds(seed_texts)
    seg_embeddings = embedder.encode_segments(seg_texts)
    R_j = embedder.seed_relevance(seg_embeddings, use_max=False, top_r=5)
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        device: str = "cpu",
        cache_dir: str = "models/",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._model = None

        # Cached matrices
        self.seed_embeddings: Optional[np.ndarray] = None   # (M, d)
        self.seed_texts: Optional[List[str]] = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"[Embedder] Loading {self.model_name} ...")
            self._model = _load_sbert(self.model_name, self.device, self.cache_dir)
            logger.info(f"[Embedder] Model loaded (dim={_embedding_dimension(self._model)})")
        return self._model

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts into embedding matrix (N, d)."""
        if not texts:
            dimension = (
                self.seed_embeddings.shape[1]
                if self.seed_embeddings is not None
                else _embedding_dimension(self.model)
            )
            return np.zeros((0, dimension), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)

    # Seed concept embedding

    def encode_seeds(self, seed_texts: List[str]) -> np.ndarray:
        """
        Encode all seed concepts and cache.

        Parameters
        ----------
        seed_texts : List[str]
            One text per concept (all aliases joined).

        Returns
        -------
        seed_embeddings : (M, d)
        """
        logger.info(f"[Embedder] Encoding {len(seed_texts)} seed concepts ...")
        self.seed_texts = seed_texts
        self.seed_embeddings = self.encode(seed_texts)
        return self.seed_embeddings

    def save_seeds(self, path: str) -> None:
        """Persist seed embeddings to disk (avoids re-encoding)."""
        with open(path, "wb") as f:
            pickle.dump({"texts": self.seed_texts, "embeddings": self.seed_embeddings}, f)
        logger.info(f"[Embedder] Seed embeddings saved -> {path}")

    def load_seeds(self, path: str) -> None:
        """Load cached seed embeddings."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.seed_texts = data["texts"]
        self.seed_embeddings = data["embeddings"]
        logger.info(f"[Embedder] Seed embeddings loaded ({self.seed_embeddings.shape})")

    # Segment embedding

    def encode_segments(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode paragraph segments.  Returns (N_segs, d).
        """
        logger.info(f"[Embedder] Encoding {len(texts)} segments ...")
        embs = self.encode(texts, show_progress=show_progress)
        logger.info(f"[Embedder] Segment embedding shape: {embs.shape}")
        return embs

    # Phrase embedding

    def encode_phrases(self, phrases: List[str]) -> np.ndarray:
        """Encode a batch of candidate phrases for Tier-2 scoring."""
        return self.encode(phrases)

    # Relevance score R_j

    def seed_relevance_scores(
        self,
        seg_embeddings: np.ndarray,  # (N, d)
        use_max: bool = False,
        top_r: int = 5,
    ) -> np.ndarray:
        """
        Compute R_j for all segments simultaneously.

        Returns
        -------
        R : (N,)  — relevance score for each segment.
        """
        assert self.seed_embeddings is not None, "Call encode_seeds() first."
        # sim_matrix[i, m] = cos(e(p_i), e(c_m))
        sim_matrix = cosine_similarity_matrix(
            seg_embeddings, self.seed_embeddings
        )  # (N, M)

        if use_max:
            return sim_matrix.max(axis=1)  # (N,)
        else:
            # Average of top-r seeds per segment
            r = min(top_r, sim_matrix.shape[1])
            sorted_sims = np.sort(sim_matrix, axis=1)[:, ::-1]  # descending
            return sorted_sims[:, :r].mean(axis=1)              # (N,)

    def seed_relevance_details(
        self,
        seg_embeddings: np.ndarray,
        top_n: int = 3,
    ) -> List[List[Tuple[str, float]]]:
        """Return nearest seed concept ids and cosine scores for each segment."""
        assert self.seed_embeddings is not None, "Call encode_seeds() first."
        concept_ids = get_concept_ids()
        sim_matrix = cosine_similarity_matrix(seg_embeddings, self.seed_embeddings)
        details: List[List[Tuple[str, float]]] = []
        for row in sim_matrix:
            indices = np.argsort(row)[::-1][:top_n]
            details.append([(concept_ids[i], float(row[i])) for i in indices])
        return details

    def phrase_seed_similarity(
        self,
        phrase_embeddings: np.ndarray,  # (P, d)
    ) -> np.ndarray:
        """
        r_jl = max_m cos(e(g_jl), e(c_m))

        Returns (P,) — max seed similarity for each phrase.
        """
        assert self.seed_embeddings is not None
        sim = cosine_similarity_matrix(phrase_embeddings, self.seed_embeddings)
        return sim.max(axis=1)

    # Coherence scoring C_j

    @staticmethod
    def coherence_scores(
        candidate_embeddings: np.ndarray,  # (K, d)
    ) -> np.ndarray:
        """
        C_j = (1 / (K-1)) Σ_{k≠j} cos(e(p_j), e(p_k))

        Mean pairwise similarity of each segment with all others in the
        candidate set.  Returns (K,).
        """
        K = candidate_embeddings.shape[0]
        if K == 1:
            return np.array([1.0])
        sim_matrix = cosine_similarity_matrix(
            candidate_embeddings, candidate_embeddings
        )  # (K, K)
        # Zero-out diagonal, sum each row, divide by (K-1)
        np.fill_diagonal(sim_matrix, 0.0)
        return sim_matrix.sum(axis=1) / (K - 1)

    @staticmethod
    def pagerank_centrality(
        candidate_embeddings: np.ndarray,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Alternative to mean C_j: PageRank centrality on the semantic graph.
        Edge weight = cosine similarity (thresholded at 0).

        Returns (K,) — centrality scores, sum = 1.
        """
        K = candidate_embeddings.shape[0]
        if K == 1:
            return np.array([1.0])

        W = cosine_similarity_matrix(candidate_embeddings, candidate_embeddings)
        np.fill_diagonal(W, 0.0)
        W = np.clip(W, 0, None)

        # Row-normalise → transition matrix
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        W = W / row_sums

        # Power iteration
        ranks = np.ones(K) / K
        for _ in range(max_iter):
            new_ranks = (1 - damping) / K + damping * W.T @ ranks
            if np.abs(new_ranks - ranks).max() < tol:
                break
            ranks = new_ranks

        return ranks
