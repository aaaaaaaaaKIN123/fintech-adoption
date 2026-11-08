"""
Train the supervised stacked FinTech adoption model.

Inputs:
    data/analysis_output/FTA_segment_features.csv
    data/analysis_input/FTA_gold_labels.csv

Outputs:
    data/analysis_output/FTA_stacked_segment_scores.csv
    data/analysis_output/FTA_stacked_panel.csv
    results/models/fta_stacked_logit.pkl
    results/logs/fta_stacked_validation.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from fintech_adoption.stacked_adoption_model import (  # noqa: E402
    aggregate_stacked_panel,
    predict_segment_probabilities,
    save_metrics,
    save_model_bundle,
    train_stacked_logit,
)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised stacked FTA logistic model.")
    parser.add_argument(
        "--features",
        default=str(ROOT / "data" / "analysis_output" / "FTA_segment_features.csv"),
        help="Segment feature table from the FTA pipeline.",
    )
    parser.add_argument(
        "--labels",
        default=str(ROOT / "data" / "analysis_input" / "FTA_gold_labels.csv"),
        help="Human gold labels with bank_id, year, seg_id, and label/y_adopted.",
    )
    parser.add_argument(
        "--segment-output",
        default=str(ROOT / "data" / "analysis_output" / "FTA_stacked_segment_scores.csv"),
        help="Output segment-level stacked probabilities.",
    )
    parser.add_argument(
        "--panel-output",
        default=str(ROOT / "data" / "analysis_output" / "FTA_stacked_panel.csv"),
        help="Output bank-year stacked panel.",
    )
    parser.add_argument(
        "--model-output",
        default=str(ROOT / "results" / "models" / "fta_stacked_logit.pkl"),
        help="Pickled model bundle output.",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(ROOT / "results" / "logs" / "fta_stacked_validation.json"),
        help="Validation metrics JSON output.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="High-confidence threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_path = _resolve_path(args.features)
    labels_path = _resolve_path(args.labels)
    if not features_path.exists():
        raise FileNotFoundError(f"Segment feature table not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Gold labels not found: {labels_path}. "
            "Create a template with data/scripts/09_create_gold_label_template.py, "
            "fill labels, then save it as FTA_gold_labels.csv."
        )

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    model, metrics = train_stacked_logit(features, labels)
    metrics.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features": str(features_path),
        "labels": str(labels_path),
        "threshold": args.threshold,
    })

    scored_segments = predict_segment_probabilities(features, model)
    panel = aggregate_stacked_panel(scored_segments, threshold=args.threshold)

    segment_output = _resolve_path(args.segment_output)
    panel_output = _resolve_path(args.panel_output)
    segment_output.parent.mkdir(parents=True, exist_ok=True)
    panel_output.parent.mkdir(parents=True, exist_ok=True)
    scored_segments.to_csv(segment_output, index=False, encoding="utf-8-sig")
    panel.to_csv(panel_output, index=False, encoding="utf-8-sig")

    save_model_bundle(model, metrics, _resolve_path(args.model_output))
    save_metrics(metrics, _resolve_path(args.metrics_output))
    print(f"[stacked-model] segments={len(scored_segments)} -> {segment_output}")
    print(f"[stacked-model] panel_rows={len(panel)} -> {panel_output}")
    print(f"[stacked-model] metrics -> {_resolve_path(args.metrics_output)}")


if __name__ == "__main__":
    main()
