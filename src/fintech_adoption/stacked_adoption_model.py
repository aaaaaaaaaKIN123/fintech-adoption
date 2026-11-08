"""
Supervised stacked FinTech adoption model.

This module trains a regularized logistic regression on human-labeled candidate
segments, then aggregates predicted adoption probabilities to bank-year
measures. It is an optional supervised layer above the weighted hybrid
baseline; the PDF scoring pipeline can run without it.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "S1_j",
    "S2_j",
    "P_llm_j",
    "llm_confidence_raw",
    "llm_adoption_probability",
    "llm_bank_is_actor",
    "llm_specific_capability",
    "llm_future_or_plan_only",
    "keyword_total_hits",
    "matched_concept_count",
    "adoption_verb_count",
    "future_plan_marker_count",
    "has_action_verb",
    "has_future_plan_marker",
    "has_bank_self_action",
    "has_specific_product_or_process",
    "keyword_density",
]

LABEL_ADOPTED = {"ADOPTED", "ADOPTED_TRUE", "TRUE", "1", "YES", "Y"}
LABEL_NOT_ADOPTED = {"DISCUSSED_ONLY", "IRRELEVANT", "NOT_ADOPTED", "FALSE", "0", "NO", "N"}


def normalize_gold_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Return labels with a binary y_adopted column."""
    required = {"bank_id", "year", "seg_id"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Gold labels missing required columns: {sorted(missing)}")

    labels = labels.copy()
    if labels.duplicated(["bank_id", "year", "seg_id"]).any():
        raise ValueError("Gold labels must have one row per bank_id, year, seg_id.")
    if "y_adopted" in labels.columns:
        values = pd.to_numeric(labels["y_adopted"], errors="raise")
        if not values.isin([0, 1]).all():
            raise ValueError("Gold y_adopted values must be 0 or 1 with no missing values.")
        labels["y_adopted"] = values.astype(int)
    elif "label" in labels.columns:
        values = labels["label"].astype(str).str.strip().str.upper()
        if not values.isin(LABEL_ADOPTED | LABEL_NOT_ADOPTED).all():
            raise ValueError("Gold labels contain blank or unknown labels; complete and review annotations first.")
        labels["y_adopted"] = values.isin(LABEL_ADOPTED).astype(int)
    else:
        raise ValueError("Gold labels need either 'label' or 'y_adopted'.")
    return labels


def prepare_training_data(features: pd.DataFrame, labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Merge features and labels, returning X and y."""
    labels = normalize_gold_labels(labels)
    merged = features.merge(
        labels[["bank_id", "year", "seg_id", "y_adopted"]],
        on=["bank_id", "year", "seg_id"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No matching rows between features and gold labels.")
    if merged["y_adopted"].nunique() < 2:
        raise ValueError("Gold labels must contain both adopted and non-adopted examples.")

    for col in FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
    X = merged[FEATURE_COLUMNS].fillna(0.0).astype(float)
    y = merged["y_adopted"].astype(int)
    return X, y


def _groups_from_features(features: pd.DataFrame) -> pd.Series:
    return features["bank_id"].astype(str) + "_" + features["year"].astype(str)


def train_stacked_logit(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[Pipeline, Dict[str, object]]:
    """Train regularized logistic regression and return metrics."""
    labels_norm = normalize_gold_labels(labels)
    merged = features.merge(
        labels_norm[["bank_id", "year", "seg_id", "y_adopted"]],
        on=["bank_id", "year", "seg_id"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No matching rows between features and gold labels.")
    if merged["y_adopted"].nunique() < 2:
        raise ValueError("Gold labels must contain both adopted and non-adopted examples.")
    for col in FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0

    X = merged[FEATURE_COLUMNS].fillna(0.0).astype(float)
    y = merged["y_adopted"].astype(int)
    groups = _groups_from_features(merged)

    model = Pipeline([
        ("scale", StandardScaler()),
        (
            "logit",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            ),
        ),
    ])

    metrics: Dict[str, object] = {
        "n_labeled_segments": int(len(merged)),
        "n_adopted": int(y.sum()),
        "n_non_adopted": int((1 - y).sum()),
        "feature_columns": FEATURE_COLUMNS,
        "model": "regularized_logistic_regression_l2",
        "group_split": "bank_id_year",
    }

    if len(set(groups)) >= 3 and min(y.value_counts()) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (pred_prob >= 0.5).astype(int)
        y_test = y.iloc[test_idx]
        metrics.update({
            "validation_rows": int(len(test_idx)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "auc": float(roc_auc_score(y_test, pred_prob)) if y_test.nunique() == 2 else None,
        })
        model.fit(X, y)
    else:
        metrics["validation_rows"] = 0
        metrics["validation_note"] = "Too few labeled groups/classes for held-out validation; fitted on all labels."
        model.fit(X, y)

    return model, metrics


def predict_segment_probabilities(features: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Add stacked_adopt_prob to segment features."""
    scored = features.copy()
    for col in FEATURE_COLUMNS:
        if col not in scored.columns:
            scored[col] = 0.0
    X = scored[FEATURE_COLUMNS].fillna(0.0).astype(float)
    scored["stacked_adopt_prob"] = model.predict_proba(X)[:, 1].round(8)
    return scored


def aggregate_stacked_panel(scored_segments: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Aggregate stacked segment probabilities to bank-year measures."""
    rows: List[Dict[str, object]] = []
    for (bank_id, year), group in scored_segments.groupby(["bank_id", "year"], sort=True):
        denominator_values = group["canonical_clean_words"].replace(0, np.nan).dropna()
        if denominator_values.empty:
            denominator_words = int(max(group.get("word_count", pd.Series([0])).sum(), 1))
        else:
            denominator_words = int(denominator_values.iloc[0])
        length_units = max(denominator_words / 10_000, 1.0)
        prob = group["stacked_adopt_prob"].astype(float)
        rows.append({
            "bank_id": bank_id,
            "year": int(year),
            "stacked_FTA_it": round(float(prob.sum() / length_units), 8),
            "stacked_AdoptDummy_it": int((prob > threshold).any()),
            "stacked_FTA_HC_it": round(float(prob[prob > threshold].sum() / length_units), 8),
            "stacked_n_candidate_segments": int(len(group)),
            "stacked_n_high_confidence_segments": int((prob > threshold).sum()),
            "canonical_clean_words": denominator_words,
            "stacked_threshold": threshold,
        })
    return pd.DataFrame(rows).sort_values(["bank_id", "year"]).reset_index(drop=True)


def save_model_bundle(model: Pipeline, metrics: Dict[str, object], path: str | Path) -> None:
    """Persist model and metadata with pickle for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": model, "metrics": metrics}, handle)


def load_model_bundle(path: str | Path) -> Dict[str, object]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_metrics(metrics: Dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
