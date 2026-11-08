"""
baseline_indices.py - Keyword and entropy baselines for FinTech adoption.

These measures are intentionally simpler than the hybrid FTA_it pipeline. They
exist to provide transparent comparison baselines in the paper:

1. keyword_FTI: keyword-frequency index normalized by canonical report length.
2. entropy_FTI: entropy-weighted index over keyword dimensions.
3. bian_Fin_adopt: Bian et al.-style narrow technology mention index.
4. hybrid_FTA_it: the main audited adoption measure, merged from aggregator.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Dict, Iterable, List

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore

from .config import AggregationConfig
from .preprocessor import BankReport

DEFAULT_FINTECH_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "input_ai_big_data": {
        "class": "input",
        "keywords": [
            "artificial intelligence",
            "machine learning",
            "big data",
            "data analytics",
            "analytics",
            "trí tuệ nhân tạo",
            "học máy",
            "dữ liệu lớn",
            "ai",
        ],
    },
    "input_blockchain": {
        "class": "input",
        "keywords": [
            "blockchain",
            "distributed ledger",
            "dlt",
            "chuỗi khối",
        ],
    },
    "input_cloud_api": {
        "class": "input",
        "keywords": [
            "cloud",
            "cloud computing",
            "đám mây",
            "api",
            "open banking",
            "open api",
            "ngân hàng mở",
        ],
    },
    "input_identity_security": {
        "class": "input",
        "keywords": [
            "ekyc",
            "e-kyc",
            "biometric",
            "biometrics",
            "digital identity",
            "định danh điện tử",
            "sinh trắc học",
        ],
    },
    "output_digital_payments": {
        "class": "output",
        "keywords": [
            "digital payment",
            "mobile payment",
            "e-wallet",
            "wallet",
            "qr",
            "vietqr",
            "thanh toán số",
            "thanh toán điện tử",
            "ví điện tử",
        ],
    },
    "output_digital_channels": {
        "class": "output",
        "keywords": [
            "mobile banking",
            "internet banking",
            "digital banking",
            "online banking",
            "ngân hàng số",
            "ngân hàng điện tử",
        ],
    },
    "output_digital_lending": {
        "class": "output",
        "keywords": [
            "digital lending",
            "online lending",
            "loan origination",
            "credit scoring",
            "cho vay trực tuyến",
            "phê duyệt khoản vay",
            "chấm điểm tín dụng",
        ],
    },
    "output_risk_operations": {
        "class": "output",
        "keywords": [
            "risk management",
            "fraud detection",
            "rpa",
            "robotic process automation",
            "automation",
            "tự động hóa",
            "quản trị rủi ro",
            "phát hiện gian lận",
        ],
    },
}

BIAN_2024_FINTECH_TAXONOMY: Dict[str, List[str]] = {
    "AI": [
        "artificial intelligence",
        "robot",
        "machine learning",
        "deep learning",
        "face recognition",
        "biometric identification",
        "voice identification",
        "intelligent risk control",
        "natural language processing",
    ],
    "Blockchain": [
        "blockchain",
        "internet of things",
        "quantum communication",
    ],
    "Cloud": [
        "cloud computing",
        "financial cloud",
        "private cloud",
        "cloud services",
        "cloud platform",
        "distributed",
    ],
    "BigData": [
        "big data analytics",
        "big data services",
        "big data technology",
        "big data models",
        "big data mining",
        "data warehousing",
        "digital marketing",
    ],
}


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_for_keyword_matching(text: str) -> str:
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_report_text(report: BankReport) -> str:
    return "\n".join(seg.text for seg in report.segments)


def _count_phrase(text_norm: str, phrase: str) -> int:
    phrase_norm = normalize_for_keyword_matching(phrase)
    if not phrase_norm:
        return 0
    pattern = r"(?<![a-z0-9])" + re.escape(phrase_norm) + r"(?![a-z0-9])"
    return len(re.findall(pattern, text_norm))


def _denominator(report: BankReport, normalization_unit: int) -> float:
    words = (
        getattr(report, "canonical_clean_words", 0)
        or getattr(report, "total_words", 0)
        or sum(seg.word_count for seg in report.segments)
    )
    return max(float(words) / normalization_unit, 1.0)


def compute_keyword_features(
    report: BankReport,
    taxonomy: Dict[str, Dict[str, List[str]]] | None = None,
    normalization_unit: int = AggregationConfig.normalization_unit,
) -> Dict[str, float | int | str]:
    """Compute keyword-frequency FinTech baseline features for one report."""
    taxonomy = taxonomy or DEFAULT_FINTECH_TAXONOMY
    text_norm = normalize_for_keyword_matching(_canonical_report_text(report))
    length_units = _denominator(report, normalization_unit)

    row: Dict[str, float | int | str] = {
        "bank_id": report.bank_id,
        "year": report.year,
    }
    total_hits = 0
    input_score = 0.0
    output_score = 0.0

    for dimension, spec in taxonomy.items():
        count = sum(_count_phrase(text_norm, keyword) for keyword in spec["keywords"])
        density = math.log1p(count) / length_units
        row[f"keyword_count_{dimension}"] = count
        row[f"keyword_dim_{dimension}"] = density
        total_hits += count
        if spec["class"] == "input":
            input_score += density
        else:
            output_score += density

    row["keyword_total_hits"] = total_hits
    row["keyword_input_technology"] = round(input_score, 8)
    row["keyword_output_application"] = round(output_score, 8)
    row["keyword_FTI"] = round(input_score + output_score, 8)
    return row

def compute_bian2024_features(
    report: BankReport,
    taxonomy: Dict[str, List[str]] | None = None,
    normalization_unit: int = AggregationConfig.normalization_unit,
) -> Dict[str, float | int | str]:
    """
    Compute Bian et al.-style technology-category mention indices.

    Bian et al. define FinTech adoption as mentions of 25 terms per 10,000
    annual-report words across AI, blockchain, cloud computing, and big data.
    This is retained as a narrow robustness/comparability measure.
    """
    taxonomy = taxonomy or BIAN_2024_FINTECH_TAXONOMY
    text_norm = normalize_for_keyword_matching(_canonical_report_text(report))
    length_units = _denominator(report, normalization_unit)

    row: Dict[str, float | int | str] = {
        "bank_id": report.bank_id,
        "year": report.year,
    }
    total_hits = 0
    total_density = 0.0
    for category, keywords in taxonomy.items():
        count = sum(_count_phrase(text_norm, keyword) for keyword in keywords)
        density = count / length_units
        row[f"bian_{category}_hits"] = int(count)
        row[f"bian_{category}_adopt"] = round(float(density), 8)
        total_hits += count
        total_density += density

    row["bian_total_hits"] = int(total_hits)
    row["bian_Fin_adopt"] = round(float(total_density), 8)
    return row


def compute_entropy_weights(
    panel: "pd.DataFrame",
    dimension_columns: Iterable[str],
) -> Dict[str, float]:
    """Compute entropy weights across keyword dimensions."""
    if not HAS_PANDAS:
        raise ImportError("pip install pandas")

    columns = list(dimension_columns)
    if not columns:
        return {}

    values = panel[columns].astype(float).copy()
    normalized = values.copy()
    for col in columns:
        col_min = values[col].min()
        col_max = values[col].max()
        if col_max > col_min:
            normalized[col] = (values[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.0

    n = max(len(normalized), 1)
    k = 1.0 / math.log(n) if n > 1 else 0.0
    diversities: Dict[str, float] = {}
    eps = 1e-12
    for col in columns:
        total = normalized[col].sum()
        if total <= 0:
            entropy = 1.0
        else:
            proportions = normalized[col] / total
            entropy = -k * sum(float(p) * math.log(float(p) + eps) for p in proportions)
            entropy = min(max(entropy, 0.0), 1.0)
        diversities[col] = 1.0 - entropy

    total_diversity = sum(diversities.values())
    if total_diversity <= 0:
        equal = 1.0 / len(columns)
        return {col: equal for col in columns}
    return {col: value / total_diversity for col, value in diversities.items()}


def _minmax_panel(panel: "pd.DataFrame", columns: List[str]) -> "pd.DataFrame":
    normalized = panel[columns].astype(float).copy()
    for col in columns:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.0
    return normalized


def build_baseline_panel(
    reports: List[BankReport],
    taxonomy: Dict[str, Dict[str, List[str]]] | None = None,
) -> "pd.DataFrame":
    """Build keyword_FTI and entropy_FTI for a list of preprocessed reports."""
    if not HAS_PANDAS:
        raise ImportError("pip install pandas")

    rows = []
    for report in reports:
        row = compute_keyword_features(report, taxonomy=taxonomy)
        row.update({
            key: value
            for key, value in compute_bian2024_features(report).items()
            if key not in {"bank_id", "year"}
        })
        rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel

    dimension_cols = [col for col in panel.columns if col.startswith("keyword_dim_")]
    weights = compute_entropy_weights(panel, dimension_cols)
    normalized_dims = _minmax_panel(panel, dimension_cols)
    panel["entropy_FTI"] = 0.0
    for col, weight in weights.items():
        panel["entropy_FTI"] += normalized_dims[col] * weight

    panel["entropy_FTI"] = panel["entropy_FTI"].round(8)
    return panel.sort_values(["bank_id", "year"]).reset_index(drop=True)


def merge_hybrid_and_baselines(
    hybrid_panel: "pd.DataFrame",
    baseline_panel: "pd.DataFrame",
) -> "pd.DataFrame":
    """Merge hybrid FTA_it with keyword and entropy baselines."""
    if not HAS_PANDAS:
        raise ImportError("pip install pandas")

    merged = hybrid_panel.merge(baseline_panel, on=["bank_id", "year"], how="left")
    if "FTA_it" in merged.columns:
        merged["hybrid_FTA_it"] = merged["FTA_it"]
    elif "hybrid_FTA_it" not in merged.columns:
        merged["hybrid_FTA_it"] = 0.0

    for col in ["keyword_FTI", "entropy_FTI", "hybrid_FTA_it"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0).astype(float).round(8)

    merged["hybrid_minus_keyword"] = (
        merged["hybrid_FTA_it"] - merged["keyword_FTI"]
    ).round(8)
    merged["hybrid_minus_entropy"] = (
        merged["hybrid_FTA_it"] - merged["entropy_FTI"]
    ).round(8)
    return merged
