import json
from pathlib import Path

import pandas as pd

from fintech_adoption.measure_comparison import (
    compare_fta_measures,
    find_disagreement_cases,
    save_measure_comparison_outputs,
)


def sample_panel():
    return pd.DataFrame(
        [
            {
                "bank_id": "A",
                "year": 2021,
                "keyword_FTI": 0.90,
                "entropy_FTI": 0.80,
                "hybrid_FTA_it": 0.10,
                "n_segments_adopted": 0,
            },
            {
                "bank_id": "B",
                "year": 2021,
                "keyword_FTI": 0.20,
                "entropy_FTI": 0.30,
                "hybrid_FTA_it": 0.75,
                "n_segments_adopted": 4,
            },
            {
                "bank_id": "C",
                "year": 2021,
                "keyword_FTI": 0.45,
                "entropy_FTI": 0.50,
                "hybrid_FTA_it": 0.48,
                "n_segments_adopted": 2,
            },
        ]
    )


def test_find_disagreement_cases_ranks_hybrid_keyword_gaps():
    cases = find_disagreement_cases(sample_panel(), top_n=2)

    assert list(cases["bank_id"]) == ["A", "B"]
    assert cases.loc[0, "primary_disagreement_type"] == "keyword_high_hybrid_low"
    assert cases.loc[1, "primary_disagreement_type"] == "hybrid_high_keyword_low"


def test_compare_fta_measures_returns_correlation_and_summary():
    result = compare_fta_measures(sample_panel(), top_n=2)

    assert {"pearson", "spearman"}.issubset(result.correlations["method"].unique())
    assert result.summary["n_observations"] == 3
    assert result.summary["n_keyword_high_hybrid_low"] == 1
    assert result.summary["n_hybrid_high_keyword_low"] == 1
    assert len(result.disagreement_cases) == 2


def test_save_measure_comparison_outputs_writes_robustness_artifacts(tmp_path):
    result = compare_fta_measures(sample_panel(), top_n=2)
    paths = save_measure_comparison_outputs(result, str(tmp_path))

    assert Path(paths["correlations"]).exists()
    assert Path(paths["disagreement_cases"]).exists()
    assert Path(paths["summary"]).exists()
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["n_observations"] == 3
