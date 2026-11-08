"""
Compare keyword_FTI, entropy_FTI, and hybrid_FTA_it.

Default input:
    data/analysis_output/FTA_panel.csv

Default outputs:
    results/robustness/FTA_measure_correlations.csv
    results/robustness/FTA_measure_disagreement_cases.csv
    results/robustness/FTA_measure_comparison_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fintech_adoption.measure_comparison import compare_fta_measures, save_measure_comparison_outputs  # noqa: E402


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare keyword, entropy, and hybrid FinTech adoption measures."
    )
    parser.add_argument(
        "--panel",
        default=str(ROOT / "data" / "analysis_output" / "FTA_panel.csv"),
        help="FTA panel CSV with keyword_FTI, entropy_FTI, hybrid_FTA_it.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "robustness"),
        help="Directory for comparison artifacts.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Number of disagreement cases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel_path = _resolve_path(args.panel)
    output_dir = _resolve_path(args.output_dir)
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel not found: {panel_path}. Run data/scripts/01_build_fta_panel.py first."
        )

    import pandas as pd

    panel = pd.read_csv(panel_path, encoding="utf-8-sig")
    result = compare_fta_measures(panel, top_n=args.top_n)
    paths = save_measure_comparison_outputs(result, str(output_dir))

    print("FTA measure comparison complete.")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
