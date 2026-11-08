"""
measure_comparison.py - Compare keyword, entropy, and hybrid FTA measures.

This module creates robustness artifacts that make the methodological
contribution visible: where keyword-based FinTech disclosure measures diverge
from the audited actual-adoption measure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore


MEASURE_COLUMNS = ["keyword_FTI", "entropy_FTI", "hybrid_FTA_it"]


@dataclass
class MeasureComparisonResult:
    correlations: "pd.DataFrame"
    disagreement_cases: "pd.DataFrame"
    summary: Dict[str, float | int]


def _require_pandas() -> None:
    if not HAS_PANDAS:
        raise ImportError("pip install pandas")


def validate_measure_panel(panel: "pd.DataFrame") -> None:
    missing = {"bank_id", "year", *MEASURE_COLUMNS} - set(panel.columns)
    if missing:
        raise ValueError(f"FTA panel missing required columns: {sorted(missing)}")


def _with_disagreement_columns(panel: "pd.DataFrame") -> "pd.DataFrame":
    df = panel.copy()
    df["hybrid_minus_keyword"] = df["hybrid_FTA_it"] - df["keyword_FTI"]
    df["hybrid_minus_entropy"] = df["hybrid_FTA_it"] - df["entropy_FTI"]
    df["abs_hybrid_keyword_gap"] = df["hybrid_minus_keyword"].abs()
    df["abs_hybrid_entropy_gap"] = df["hybrid_minus_entropy"].abs()
    df["max_abs_gap"] = df[["abs_hybrid_keyword_gap", "abs_hybrid_entropy_gap"]].max(axis=1)

    keyword_median = df["keyword_FTI"].median()
    hybrid_median = df["hybrid_FTA_it"].median()

    def classify(row):
        if row["keyword_FTI"] > keyword_median and row["hybrid_FTA_it"] < hybrid_median:
            return "keyword_high_hybrid_low"
        if row["hybrid_FTA_it"] > hybrid_median and row["keyword_FTI"] < keyword_median:
            return "hybrid_high_keyword_low"
        if row["entropy_FTI"] > row["hybrid_FTA_it"]:
            return "entropy_above_hybrid"
        if row["hybrid_FTA_it"] > row["entropy_FTI"]:
            return "hybrid_above_entropy"
        return "aligned"

    df["primary_disagreement_type"] = df.apply(classify, axis=1)
    return df


def find_disagreement_cases(panel: "pd.DataFrame", top_n: int = 20) -> "pd.DataFrame":
    """Return the largest measure-disagreement cases for audit review."""
    _require_pandas()
    validate_measure_panel(panel)
    df = _with_disagreement_columns(panel)
    useful_cols: List[str] = [
        "bank_id",
        "year",
        "keyword_FTI",
        "entropy_FTI",
        "hybrid_FTA_it",
        "hybrid_minus_keyword",
        "hybrid_minus_entropy",
        "abs_hybrid_keyword_gap",
        "abs_hybrid_entropy_gap",
        "max_abs_gap",
        "primary_disagreement_type",
    ]
    optional_cols = [
        "FTA_it",
        "FTA_binary",
        "n_segments_adopted",
        "n_segments_discussed",
        "top_key_phrases",
    ]
    for col in optional_cols:
        if col in df.columns:
            useful_cols.append(col)

    return (
        df.sort_values(["max_abs_gap", "bank_id", "year"], ascending=[False, True, True])
        .head(top_n)
        .loc[:, useful_cols]
        .reset_index(drop=True)
    )


def compute_correlations(panel: "pd.DataFrame") -> "pd.DataFrame":
    """Compute Pearson and Spearman correlations among the three measures."""
    _require_pandas()
    validate_measure_panel(panel)
    rows = []
    for method in ["pearson", "spearman"]:
        corr = panel[MEASURE_COLUMNS].corr(method=method)
        for left in MEASURE_COLUMNS:
            for right in MEASURE_COLUMNS:
                if left >= right:
                    continue
                rows.append(
                    {
                        "method": method,
                        "measure_left": left,
                        "measure_right": right,
                        "correlation": round(float(corr.loc[left, right]), 8),
                    }
                )
    return pd.DataFrame(rows)


def compare_fta_measures(
    panel: "pd.DataFrame",
    top_n: int = 20,
) -> MeasureComparisonResult:
    """Build comparison artifacts for keyword, entropy, and hybrid measures."""
    _require_pandas()
    validate_measure_panel(panel)
    df = _with_disagreement_columns(panel)
    correlations = compute_correlations(df)
    disagreement_cases = find_disagreement_cases(df, top_n=top_n)

    keyword_median = df["keyword_FTI"].median()
    hybrid_median = df["hybrid_FTA_it"].median()
    summary: Dict[str, float | int] = {
        "n_observations": int(len(df)),
        "mean_keyword_FTI": round(float(df["keyword_FTI"].mean()), 8),
        "mean_entropy_FTI": round(float(df["entropy_FTI"].mean()), 8),
        "mean_hybrid_FTA_it": round(float(df["hybrid_FTA_it"].mean()), 8),
        "mean_abs_hybrid_keyword_gap": round(float(df["abs_hybrid_keyword_gap"].mean()), 8),
        "mean_abs_hybrid_entropy_gap": round(float(df["abs_hybrid_entropy_gap"].mean()), 8),
        "n_keyword_high_hybrid_low": int(
            ((df["keyword_FTI"] > keyword_median) & (df["hybrid_FTA_it"] < hybrid_median)).sum()
        ),
        "n_hybrid_high_keyword_low": int(
            ((df["hybrid_FTA_it"] > hybrid_median) & (df["keyword_FTI"] < keyword_median)).sum()
        ),
    }
    return MeasureComparisonResult(
        correlations=correlations,
        disagreement_cases=disagreement_cases,
        summary=summary,
    )


def save_measure_comparison_outputs(
    result: MeasureComparisonResult,
    output_dir: str,
) -> Dict[str, str]:
    """Write robustness artifacts and return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "correlations": str(out / "FTA_measure_correlations.csv"),
        "disagreement_cases": str(out / "FTA_measure_disagreement_cases.csv"),
        "summary": str(out / "FTA_measure_comparison_summary.json"),
    }
    result.correlations.to_csv(paths["correlations"], index=False, encoding="utf-8-sig")
    result.disagreement_cases.to_csv(paths["disagreement_cases"], index=False, encoding="utf-8-sig")
    with open(paths["summary"], "w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, ensure_ascii=False, indent=2)
    return paths
