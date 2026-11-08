"""
Build segment-level feature tables for FinTech adoption stacking.

The table is produced from scored candidate segments and is the handoff point
for gold annotation, supervised logistic stacking, and audit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore

from .preprocessor import BankReport, Segment

SEGMENT_FEATURE_COLUMNS = [
    "bank_id",
    "year",
    "seg_id",
    "page_num",
    "word_count",
    "canonical_clean_words",
    "language",
    "S1_j",
    "S2_j",
    "P_llm_j",
    "S_j",
    "R_j",
    "C_j",
    "llm_label",
    "llm_confidence_raw",
    "llm_fintech_type",
    "llm_evidence",
    "llm_adoption_probability",
    "llm_adoption_stage",
    "llm_bank_is_actor",
    "llm_specific_capability",
    "llm_future_or_plan_only",
    "nearest_seed_ids",
    "nearest_seed_scores",
    "key_phrases",
    "keyword_total_hits",
    "matched_concept_count",
    "matched_concept_ids",
    "adoption_verb_count",
    "future_plan_marker_count",
    "has_action_verb",
    "has_future_plan_marker",
    "has_bank_self_action",
    "has_specific_product_or_process",
    "keyword_density",
    "text_preview",
]


def _bool_to_int(value: object) -> int:
    return int(bool(value))


def _candidate_segments(report: BankReport) -> Iterable[Segment]:
    for segment in report.segments:
        if segment.S1_j > 0 or segment.S2_j > 0 or segment.P_llm_j > 0 or segment.S_j > 0:
            yield segment


def build_segment_feature_table(reports: List[BankReport]) -> "pd.DataFrame":
    """Return one row per scored candidate segment."""
    if not HAS_PANDAS:
        raise ImportError("pip install pandas")

    rows = []
    for report in reports:
        denominator_words = int(
            getattr(report, "canonical_clean_words", 0)
            or getattr(report, "total_words", 0)
            or sum(seg.word_count for seg in report.segments)
        )
        for seg in _candidate_segments(report):
            keyword = seg.keyword_features or {}
            rows.append({
                "bank_id": report.bank_id,
                "year": report.year,
                "seg_id": seg.seg_id,
                "page_num": seg.page_num,
                "word_count": seg.word_count,
                "canonical_clean_words": denominator_words,
                "language": seg.language,
                "S1_j": seg.S1_j,
                "S2_j": seg.S2_j,
                "P_llm_j": seg.P_llm_j,
                "S_j": seg.S_j,
                "R_j": seg.R_j,
                "C_j": seg.C_j,
                "llm_label": seg.llm_label,
                "llm_confidence_raw": seg.llm_confidence_raw,
                "llm_fintech_type": seg.llm_fintech_type,
                "llm_evidence": seg.llm_evidence,
                "llm_adoption_probability": seg.llm_adoption_probability,
                "llm_adoption_stage": seg.llm_adoption_stage,
                "llm_bank_is_actor": _bool_to_int(seg.llm_bank_is_actor),
                "llm_specific_capability": _bool_to_int(seg.llm_specific_capability),
                "llm_future_or_plan_only": _bool_to_int(seg.llm_future_or_plan_only),
                "nearest_seed_ids": " | ".join(seg.nearest_seed_ids),
                "nearest_seed_scores": " | ".join(f"{score:.4f}" for score in seg.nearest_seed_scores),
                "key_phrases": " | ".join(seg.key_phrases),
                "keyword_total_hits": int(keyword.get("keyword_total_hits", 0)),
                "matched_concept_count": int(keyword.get("matched_concept_count", 0)),
                "matched_concept_ids": " | ".join(keyword.get("matched_concept_ids", []) or []),
                "adoption_verb_count": int(keyword.get("adoption_verb_count", 0)),
                "future_plan_marker_count": int(keyword.get("future_plan_marker_count", 0)),
                "has_action_verb": _bool_to_int(keyword.get("has_action_verb", False)),
                "has_future_plan_marker": _bool_to_int(keyword.get("has_future_plan_marker", False)),
                "has_bank_self_action": _bool_to_int(keyword.get("has_bank_self_action", False)),
                "has_specific_product_or_process": _bool_to_int(
                    keyword.get("has_specific_product_or_process", False)
                ),
                "keyword_density": float(keyword.get("keyword_density", 0.0)),
                "text_preview": seg.text[:500],
            })

    if not rows:
        return pd.DataFrame(columns=SEGMENT_FEATURE_COLUMNS)
    return (
        pd.DataFrame(rows, columns=SEGMENT_FEATURE_COLUMNS)
        .sort_values(["bank_id", "year", "seg_id"])
        .reset_index(drop=True)
    )


def save_segment_feature_table(reports: List[BankReport], output_dir: str) -> str:
    """Save candidate segment features for annotation and stacking."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "FTA_segment_features.csv")
    build_segment_feature_table(reports).to_csv(path, index=False, encoding="utf-8-sig")
    return path
