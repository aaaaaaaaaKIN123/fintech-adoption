import math

import pandas as pd

from fintech_adoption.baseline_indices import (
    build_baseline_panel,
    compute_bian2024_features,
    compute_entropy_weights,
    compute_keyword_features,
    merge_hybrid_and_baselines,
)
from fintech_adoption.preprocessor import BankReport, Segment


def make_report(bank_id, year, text, canonical_words=10_000):
    return BankReport(
        bank_id=bank_id,
        year=year,
        pdf_path="fake.pdf",
        segments=[Segment(seg_id=0, text=text, page_num=1, char_offset=0)],
        clean_total_words=canonical_words,
        canonical_clean_words=canonical_words,
    )


def test_keyword_fti_counts_input_and_output_fintech_terms_with_length_normalization():
    report = make_report(
        "VCB",
        2023,
        "The bank launched eKYC, mobile banking, cloud API and digital payment services.",
        canonical_words=20_000,
    )

    features = compute_keyword_features(report)

    assert features["keyword_total_hits"] >= 4
    assert features["keyword_input_technology"] > 0
    assert features["keyword_output_application"] > 0
    assert features["keyword_FTI"] > 0
    assert features["keyword_FTI"] < features["keyword_total_hits"]


def test_entropy_fti_uses_cross_report_dimension_dispersion():
    reports = [
        make_report("A", 2021, "eKYC eKYC cloud API blockchain"),
        make_report("B", 2021, "mobile banking digital payment VietQR online lending"),
        make_report("C", 2021, "ordinary branch operations and balance sheet discussion"),
    ]

    panel = build_baseline_panel(reports)
    dimension_cols = [col for col in panel.columns if col.startswith("keyword_dim_")]
    weights = compute_entropy_weights(panel, dimension_cols)

    assert "keyword_FTI" in panel.columns
    assert "entropy_FTI" in panel.columns
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)
    assert panel.loc[panel["bank_id"] == "A", "entropy_FTI"].iloc[0] > 0
    assert panel.loc[panel["bank_id"] == "C", "entropy_FTI"].iloc[0] == 0


def test_bian2024_features_count_category_mentions_per_10000_words():
    report = make_report(
        "TCB",
        2024,
        (
            "Artificial intelligence and machine learning support intelligent risk control. "
            "The bank expanded blockchain pilots, cloud computing, private cloud, "
            "big data analytics, data warehousing, and digital marketing."
        ),
        canonical_words=20_000,
    )

    features = compute_bian2024_features(report)

    assert features["bian_AI_hits"] == 3
    assert features["bian_Blockchain_hits"] == 1
    assert features["bian_Cloud_hits"] == 2
    assert features["bian_BigData_hits"] == 3
    assert features["bian_total_hits"] == 9
    assert features["bian_Fin_adopt"] == 4.5


def test_build_baseline_panel_includes_bian2024_robustness_columns():
    reports = [
        make_report("A", 2022, "cloud computing big data analytics"),
        make_report("B", 2022, "ordinary annual report text"),
    ]

    panel = build_baseline_panel(reports)

    assert "bian_total_hits" in panel.columns
    assert "bian_Fin_adopt" in panel.columns
    assert "bian_AI_adopt" in panel.columns
    assert "bian_Blockchain_adopt" in panel.columns
    assert "bian_Cloud_adopt" in panel.columns
    assert "bian_BigData_adopt" in panel.columns
    assert panel.loc[panel["bank_id"] == "A", "bian_total_hits"].iloc[0] == 2
    assert panel.loc[panel["bank_id"] == "B", "bian_Fin_adopt"].iloc[0] == 0


def test_merge_hybrid_and_baseline_panel_keeps_three_parallel_measures():
    hybrid = pd.DataFrame(
        [{"bank_id": "VCB", "year": 2023, "FTA_it": 0.72, "n_segments_adopted": 3}]
    )
    baseline = pd.DataFrame(
        [{"bank_id": "VCB", "year": 2023, "keyword_FTI": 0.40, "entropy_FTI": 0.55}]
    )

    merged = merge_hybrid_and_baselines(hybrid, baseline)

    assert merged.loc[0, "hybrid_FTA_it"] == 0.72
    assert merged.loc[0, "keyword_FTI"] == 0.40
    assert merged.loc[0, "entropy_FTI"] == 0.55
    assert merged.loc[0, "hybrid_minus_keyword"] == 0.32
    assert merged.loc[0, "hybrid_minus_entropy"] == 0.17
