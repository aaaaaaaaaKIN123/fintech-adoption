"""
Create a gold-label annotation template from FTA candidate segment features.

Input:
    data/analysis_output/FTA_segment_features.csv

Output:
    data/analysis_input/FTA_gold_labels_template.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def create_gold_label_template(
    features: pd.DataFrame,
    per_bank_year: int = 5,
    limit: int | None = None,
) -> pd.DataFrame:
    """Select high-signal candidate segments for human annotation."""
    required = {"bank_id", "year", "seg_id", "S_j", "text_preview"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature table missing required columns: {sorted(missing)}")

    ranked = features.sort_values(["bank_id", "year", "S_j"], ascending=[True, True, False])
    selected = ranked.groupby(["bank_id", "year"], group_keys=False).head(per_bank_year)
    selected = selected.sort_values(["bank_id", "year", "S_j"], ascending=[True, True, False])
    if limit is not None:
        selected = selected.head(limit)

    columns = [
        "bank_id",
        "year",
        "seg_id",
        "page_num",
        "llm_label",
        "llm_confidence_raw",
        "S1_j",
        "S2_j",
        "P_llm_j",
        "S_j",
        "matched_concept_ids",
        "key_phrases",
        "text_preview",
    ]
    available = [col for col in columns if col in selected.columns]
    template = selected[available].copy()
    template["label"] = ""
    template["annotator_id"] = ""
    template["annotation_notes"] = ""
    return template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FTA gold-label annotation template.")
    parser.add_argument(
        "--features",
        default=str(ROOT / "data" / "analysis_output" / "FTA_segment_features.csv"),
        help="Segment feature table from the FTA pipeline.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "analysis_input" / "FTA_gold_labels_template.csv"),
        help="Annotation template output CSV.",
    )
    parser.add_argument("--per-bank-year", type=int, default=5, help="Rows to sample per bank-year.")
    parser.add_argument("--limit", type=int, default=None, help="Optional total row cap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_path = _resolve_path(args.features)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(features_path)
    template = create_gold_label_template(features, args.per_bank_year, args.limit)
    template.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[gold-label-template] rows={len(template)} -> {output_path}")


if __name__ == "__main__":
    main()
