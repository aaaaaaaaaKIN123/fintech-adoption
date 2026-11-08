"""
aggregator.py - Final segment scoring and bank-year FTA aggregation.

FTA_it is normalized by canonical report length after bilingual duplicate
removal. This avoids double-counting annual reports that contain Vietnamese and
English versions of the same disclosure in one PDF.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore

from .config import AggregationConfig
from .preprocessor import BankReport, Segment

logger = logging.getLogger(__name__)


@dataclass
class BankYearRecord:
    """One bank-year observation in the final FTA panel."""

    bank_id: str
    year: int
    FTA_it: float = 0.0
    FTA_binary: int = 0

    avg_S1: float = 0.0
    avg_S2: float = 0.0
    avg_P_llm: float = 0.0

    n_segments_total: int = 0
    n_candidate_segments: int = 0
    n_scored_segments: int = 0
    n_segments_adopted: int = 0
    n_segments_discussed: int = 0

    total_words: int = 0
    clean_total_words: int = 0
    raw_total_words: int = 0
    canonical_clean_words: int = 0
    retained_segment_words: int = 0
    num_pages: int = 0

    domain_scores: dict = field(default_factory=dict)
    top_key_phrases: List[str] = field(default_factory=list)


def compute_segment_final_score(
    seg: Segment,
    cfg: AggregationConfig,
    run_llm: bool = True,
) -> float:
    """
    S_j = 1{LLM label is ADOPTED} * (beta1*S1_j + beta2*S2_j + beta3*P_llm_j).

    If LLM is disabled, the Tier-1 and Tier-2 weights are renormalized.
    """
    if run_llm:
        if seg.llm_label != "ADOPTED":
            score = 0.0
        else:
            score = cfg.beta1 * seg.S1_j + cfg.beta2 * seg.S2_j + cfg.beta3 * seg.P_llm_j
    else:
        denom = cfg.beta1 + cfg.beta2
        score = (cfg.beta1 * seg.S1_j + cfg.beta2 * seg.S2_j) / max(denom, 1e-9)

    seg.S_j = float(score)
    return seg.S_j


def compute_all_final_scores(
    segments: List[Segment],
    cfg: AggregationConfig,
    run_llm: bool = True,
) -> None:
    """Write final segment scores in-place."""
    for seg in segments:
        compute_segment_final_score(seg, cfg, run_llm)


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _report_denominator_words(report: BankReport) -> int:
    return int(
        getattr(report, "canonical_clean_words", 0)
        or getattr(report, "total_words", 0)
        or sum(seg.word_count for seg in report.segments)
    )


def aggregate_report(
    report: BankReport,
    top_k_segments: List[Segment],
    cfg: AggregationConfig,
    run_llm: bool = True,
    binary_threshold: float = 0.10,
) -> BankYearRecord:
    """
    Produce one BankYearRecord from a processed report.

    FTA_it = sum_j(S_j) / L_it, where L_it is canonical_clean_words / 10,000.
    """
    compute_all_final_scores(top_k_segments, cfg, run_llm)

    scored_segs = [seg for seg in top_k_segments if seg.S_j > cfg.min_score_threshold]
    denominator_words = _report_denominator_words(report)
    L_it = max(denominator_words / cfg.normalization_unit, 1.0)

    total_score = sum(seg.S_j for seg in scored_segs)
    FTA_it = total_score / L_it

    n_adopted = sum(1 for seg in top_k_segments if seg.llm_label == "ADOPTED")
    n_discussed = sum(1 for seg in top_k_segments if seg.llm_label == "DISCUSSED_ONLY")

    all_phrases: List[str] = []
    for seg in sorted(scored_segs, key=lambda item: item.S_j, reverse=True)[:10]:
        all_phrases.extend(seg.key_phrases)

    seen = set()
    top_phrases = []
    for phrase in all_phrases:
        if phrase not in seen and len(phrase) > 3:
            seen.add(phrase)
            top_phrases.append(phrase)

    record = BankYearRecord(
        bank_id=report.bank_id,
        year=report.year,
        FTA_it=round(FTA_it, 6),
        FTA_binary=int(FTA_it > binary_threshold),
        avg_S1=round(_safe_mean([seg.S1_j for seg in scored_segs]), 4),
        avg_S2=round(_safe_mean([seg.S2_j for seg in scored_segs]), 4),
        avg_P_llm=round(_safe_mean([seg.P_llm_j for seg in scored_segs]), 4)
        if run_llm
        else 0.0,
        n_segments_total=len(report.segments),
        n_candidate_segments=len(top_k_segments),
        n_scored_segments=len(scored_segs),
        n_segments_adopted=n_adopted,
        n_segments_discussed=n_discussed,
        total_words=denominator_words,
        clean_total_words=int(getattr(report, "clean_total_words", denominator_words)),
        raw_total_words=int(getattr(report, "raw_total_words", denominator_words)),
        canonical_clean_words=denominator_words,
        retained_segment_words=int(
            getattr(report, "retained_segment_words", 0)
            or sum(seg.word_count for seg in report.segments)
        ),
        num_pages=report.metadata.get("num_pages", 0),
        top_key_phrases=top_phrases[:15],
    )

    logger.info(
        "[Aggregator] %s %s: FTA=%.4f | ADOPTED=%s | DISCUSSED=%s | "
        "canonical_words=%s",
        report.bank_id,
        report.year,
        FTA_it,
        n_adopted,
        n_discussed,
        f"{denominator_words:,}",
    )
    return record


def build_panel(records: List[BankYearRecord]) -> "pd.DataFrame":
    """Convert records to a pandas panel DataFrame."""
    if not HAS_PANDAS:
        raise ImportError("pip install pandas")

    rows = []
    for record in records:
        row = {key: value for key, value in asdict(record).items() if key != "domain_scores"}
        row["top_key_phrases"] = " | ".join(record.top_key_phrases)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["bank_id", "year"]).reset_index(drop=True)
    if not df.empty and df["FTA_it"].max() > 0:
        df["FTA_it_norm"] = df["FTA_it"] / df["FTA_it"].max()
    else:
        df["FTA_it_norm"] = 0.0
    return df


def save_panel_csv(df: "pd.DataFrame", output_dir: str = "outputs/") -> str:
    """Save the FTA panel to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "FTA_panel.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("[Aggregator] Panel saved -> %s", path)
    return path


def save_records_json(records: List[BankYearRecord], output_dir: str = "outputs/") -> str:
    """Save full bank-year records as JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "FTA_records.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, ensure_ascii=False, indent=2)
    logger.info("[Aggregator] Records saved -> %s", path)
    return path


def save_segment_detail(
    bank_id: str,
    year: int,
    segments: List[Segment],
    output_dir: str = "outputs/",
) -> str:
    """Save segment-level audit trail for one report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"{bank_id}_{year}_segments.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for seg in segments:
            row = {
                "bank_id": bank_id,
                "year": year,
                "seg_id": seg.seg_id,
                "page_num": seg.page_num,
                "char_offset": seg.char_offset,
                "word_count": seg.word_count,
                "language": seg.language,
                "canonical_seg_id": seg.canonical_seg_id,
                "duplicate_seg_ids": seg.duplicate_seg_ids,
                "languages_in_cluster": seg.languages_in_cluster,
                "duplicate_similarity": seg.duplicate_similarity,
                "tfidf_doc_index": seg.tfidf_doc_index,
                "nearest_seed_ids": seg.nearest_seed_ids,
                "nearest_seed_scores": seg.nearest_seed_scores,
                "R_j": seg.R_j,
                "C_j": seg.C_j,
                "S1_j": seg.S1_j,
                "S2_j": seg.S2_j,
                "llm_label": seg.llm_label,
                "llm_confidence_raw": seg.llm_confidence_raw,
                "llm_fintech_type": seg.llm_fintech_type,
                "llm_evidence": seg.llm_evidence,
                "llm_adoption_probability": seg.llm_adoption_probability,
                "llm_adoption_stage": seg.llm_adoption_stage,
                "llm_bank_is_actor": seg.llm_bank_is_actor,
                "llm_specific_capability": seg.llm_specific_capability,
                "llm_future_or_plan_only": seg.llm_future_or_plan_only,
                "llm_reason": seg.llm_reason,
                "llm_raw_response": seg.llm_raw_response,
                "P_llm_j": seg.P_llm_j,
                "S_j": seg.S_j,
                "key_phrases": seg.key_phrases,
                "keyword_features": seg.keyword_features,
                "text_preview": seg.text[:200],
                "text": seg.text,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("[Aggregator] Segment detail saved -> %s", path)
    return path
