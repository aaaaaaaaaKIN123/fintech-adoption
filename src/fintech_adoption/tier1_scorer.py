"""
tier1_scorer.py — Tier-1 paragraph-level scoring.

Implements
──────────
  For each segment p_j:
    R_j   = seed relevance  (avg top-r or max cosine)
    C_j   = internal coherence with top-K peers
    S_j^(1) = α·R_j + (1-α)·C_j

  Output: ranked list of top-K candidate segments.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .preprocessor import BankReport, Segment
from .embedder import FinTechEmbedder
from .config import Tier1Config

logger = logging.getLogger(__name__)


def score_tier1(
    report: BankReport,
    seg_embeddings: np.ndarray,       # (N_segs, d)  pre-computed
    embedder: FinTechEmbedder,
    cfg: Tier1Config,
) -> Tuple[List[Segment], np.ndarray]:
    """
    Run Tier-1 scoring on all segments of a single BankReport.

    Parameters
    ----------
    report         : BankReport with populated segments list.
    seg_embeddings : Pre-computed embedding matrix (N_segs, d).
    embedder       : FinTechEmbedder with seed embeddings already loaded.
    cfg            : Tier1Config hyperparameters.

    Returns
    -------
    top_k_segments : List[Segment] — top-K segments sorted by S^(1) desc.
    top_k_embs     : np.ndarray    — corresponding embeddings (K, d).
    """
    N = len(report.segments)
    if N == 0:
        logger.warning(f"[Tier1] {report.bank_id} {report.year}: no segments.")
        return [], np.zeros((0, seg_embeddings.shape[1]))

    assert seg_embeddings.shape[0] == N, (
        f"Embedding rows ({seg_embeddings.shape[0]}) ≠ segments ({N})"
    )

    logger.info(f"[Tier1] {report.bank_id} {report.year}: scoring {N} segments ...")

    # Step 1: Relevance score R_j
    R = embedder.seed_relevance_scores(
        seg_embeddings,
        use_max=cfg.use_max,
        top_r=cfg.top_r_seeds,
    )  # (N,)
    seed_details = embedder.seed_relevance_details(seg_embeddings, top_n=3)

    # Step 2: Select top-K initial candidates
    K = min(cfg.top_k, N)
    top_k_indices = np.argsort(R)[::-1][:K]

    top_k_embs = seg_embeddings[top_k_indices]  # (K, d)

    # Step 3: Coherence score C_j within the top-K set
    if cfg.use_graph_centrality:
        C_topk = embedder.pagerank_centrality(
            top_k_embs, damping=cfg.pagerank_damping
        )
        # Normalise PageRank to [0, 1]
        c_min, c_max = C_topk.min(), C_topk.max()
        C_topk = (C_topk - c_min) / (c_max - c_min + 1e-9)
    else:
        C_topk = embedder.coherence_scores(top_k_embs)  # (K,)

    # Step 4: Composite Tier-1 score S_j^(1)
    α = cfg.alpha
    R_topk = R[top_k_indices]
    S1_topk = α * R_topk + (1 - α) * C_topk  # (K,)

    # Step 5: Write scores back to Segment objects
    for rank, (idx, s1, rj, cj) in enumerate(
        zip(top_k_indices, S1_topk, R_topk, C_topk)
    ):
        seg = report.segments[idx]
        seg.R_j = float(rj)
        seg.C_j = float(cj)
        seg.S1_j = float(s1)
        details = seed_details[idx]
        seg.nearest_seed_ids = [seed_id for seed_id, _score in details]
        seg.nearest_seed_scores = [float(score) for _seed_id, score in details]

    # Step 6: Sort top-K by S^(1) descending
    ranked_order = np.argsort(S1_topk)[::-1]
    top_k_segments = [report.segments[top_k_indices[i]] for i in ranked_order]
    top_k_embs_sorted = top_k_embs[ranked_order]

    logger.info(
        f"[Tier1] Top-{K} selected | "
        f"S1 range [{S1_topk.min():.3f}, {S1_topk.max():.3f}] | "
        f"Avg R={R_topk.mean():.3f}, Avg C={C_topk.mean():.3f}"
    )

    return top_k_segments, top_k_embs_sorted


def score_tier1_batch(
    reports: List[BankReport],
    all_seg_embeddings: List[np.ndarray],
    embedder: FinTechEmbedder,
    cfg: Tier1Config,
) -> List[Tuple[List[Segment], np.ndarray]]:
    """
    Batch version: process a list of reports in sequence.

    Returns
    -------
    List of (top_k_segments, top_k_embs) per report.
    """
    results = []
    for report, embs in zip(reports, all_seg_embeddings):
        result = score_tier1(report, embs, embedder, cfg)
        results.append(result)
    return results
