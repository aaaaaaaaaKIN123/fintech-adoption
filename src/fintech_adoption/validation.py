"""
validation.py — Calibration, evaluation, and hyperparameter selection.

Responsibilities
────────────────
  1. Load manually annotated gold-standard segments.
  2. Compute Precision, Recall, F1 for the full pipeline.
  3. Grid-search over (α, K, β1, β2, β3) to maximise F1.
  4. Report inter-rater reliability metrics (Cohen's κ).
  5. Validate construct validity (correlation with observable proxies).

Gold standard format (CSV columns):
  bank_id, year, seg_id, page_num, text, true_label
  where true_label ∈ {ADOPTED, DISCUSSED_ONLY, IRRELEVANT}

Usage
-----
  validator = PipelineValidator(gold_csv="data/gold_standard.csv")
  validator.evaluate_pipeline(scored_segments)
  validator.grid_search_alpha(...)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .preprocessor import Segment
from .config import AggregationConfig

logger = logging.getLogger(__name__)



# METRICS


def precision_recall_f1(
    y_true: List[str],
    y_pred: List[str],
    positive_class: str = "ADOPTED",
) -> Tuple[float, float, float]:
    """Binary P/R/F1 treating `positive_class` as positive."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def cohens_kappa(y1: List[str], y2: List[str], labels: Optional[List[str]] = None) -> float:
    """
    Compute Cohen's κ for inter-rater reliability between two annotators.
    """
    if labels is None:
        labels = sorted(set(y1) | set(y2))

    label_idx = {l: i for i, l in enumerate(labels)}
    n = len(y1)
    K = len(labels)
    conf = np.zeros((K, K))

    for a, b in zip(y1, y2):
        if a in label_idx and b in label_idx:
            conf[label_idx[a], label_idx[b]] += 1

    p_observed = conf.diagonal().sum() / n
    p_expected = (conf.sum(axis=1) / n) @ (conf.sum(axis=0) / n)
    kappa = (p_observed - p_expected) / (1 - p_expected + 1e-9)
    return float(kappa)


def classification_report(y_true: List[str], y_pred: List[str]) -> str:
    """Simple text classification report."""
    all_labels = sorted(set(y_true) | set(y_pred))
    lines = [f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"]
    lines.append("-" * 60)
    for label in all_labels:
        p, r, f1 = precision_recall_f1(y_true, y_pred, positive_class=label)
        support = y_true.count(label)
        lines.append(f"{label:<20} {p:>10.3f} {r:>10.3f} {f1:>10.3f} {support:>10}")
    lines.append("-" * 60)
    overall_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    lines.append(f"{'Accuracy':<20} {overall_acc:>10.3f}")
    return "\n".join(lines)

def save_validation_report(metrics: Dict, output_path: str) -> str:
    """Persist validation metrics as JSON for reproducibility."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    logger.info("[Validation] Report saved -> %s", output_path)
    return output_path



# GOLD STANDARD LOADER


@dataclass
class GoldRecord:
    bank_id: str
    year: int
    seg_id: int
    text: str
    true_label: str  # ADOPTED | DISCUSSED_ONLY | IRRELEVANT


def load_gold_standard(csv_path: str) -> List[GoldRecord]:
    """Load manually annotated gold standard from CSV."""
    if not HAS_PANDAS:
        raise ImportError("pip install pandas")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"bank_id", "year", "seg_id", "text", "true_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gold CSV missing columns: {missing}")
    records = []
    for _, row in df.iterrows():
        records.append(GoldRecord(
            bank_id=str(row["bank_id"]),
            year=int(row["year"]),
            seg_id=int(row["seg_id"]),
            text=str(row["text"]),
            true_label=str(row["true_label"]).upper(),
        ))
    logger.info(f"[Validation] Loaded {len(records)} gold-standard segments.")
    return records



# PIPELINE VALIDATOR


class PipelineValidator:
    """
    Evaluates the pipeline against a gold-standard annotated dataset.

    Workflow
    ────────
    1. Match pipeline output segments to gold records by (bank_id, year, seg_id).
    2. Compute precision/recall/F1 for ADOPTED classification.
    3. Optionally grid-search hyperparameters.
    """

    def __init__(self, gold_records: Optional[List[GoldRecord]] = None, gold_csv: Optional[str] = None):
        if gold_csv:
            self.gold_records = load_gold_standard(gold_csv)
        elif gold_records:
            self.gold_records = gold_records
        else:
            self.gold_records = []
        # Index by (bank_id, year, seg_id)
        self.gold_index: Dict[Tuple, str] = {
            (r.bank_id, r.year, r.seg_id): r.true_label
            for r in self.gold_records
        }

    def evaluate_pipeline(
        self,
        scored_segments: List[Segment],
        bank_id: str,
        year: int,
    ) -> Dict[str, float]:
        """
        Evaluate Tier-1+2+LLM pipeline against gold labels for one report.

        Returns dict with precision, recall, f1, kappa.
        """
        y_true, y_pred = [], []
        for seg in scored_segments:
            key = (bank_id, year, seg.seg_id)
            if key in self.gold_index:
                y_true.append(self.gold_index[key])
                pred = seg.llm_label if seg.llm_label else self._threshold_label(seg)
                y_pred.append(pred)

        if not y_true:
            logger.warning(f"[Validation] No matching gold records for {bank_id} {year}.")
            return {}

        p, r, f1 = precision_recall_f1(y_true, y_pred)
        kappa = cohens_kappa(y_true, y_pred)

        results = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "kappa": round(kappa, 4),
            "n_evaluated": len(y_true),
        }
        logger.info(f"[Validation] {bank_id} {year}: {results}")
        print(classification_report(y_true, y_pred))
        return results

    def _threshold_label(self, seg: Segment, threshold: float = 0.4) -> str:
        """Fallback label from S1 score when LLM not run."""
        if seg.S1_j >= threshold:
            return "ADOPTED"
        elif seg.S1_j >= 0.2:
            return "DISCUSSED_ONLY"
        return "IRRELEVANT"

    # Hyperparameter grid search

    def grid_search_alpha(
        self,
        scored_segments: List[Segment],
        bank_id: str,
        year: int,
        alpha_values: Optional[List[float]] = None,
    ) -> Tuple[float, Dict]:
        """
        Grid-search over α ∈ [0, 1] to maximise F1.
        Requires that R_j and C_j are pre-computed on segments.
        """
        if alpha_values is None:
            alpha_values = [round(v, 2) for v in np.arange(0.1, 1.0, 0.1)]

        best_alpha, best_f1 = 0.5, 0.0
        results_table = []

        for alpha in alpha_values:
            # Temporarily update S1 scores
            for seg in scored_segments:
                seg.S1_j = alpha * seg.R_j + (1 - alpha) * seg.C_j
            metrics = self.evaluate_pipeline(scored_segments, bank_id, year)
            f1 = metrics.get("f1", 0.0)
            results_table.append({"alpha": alpha, **metrics})
            if f1 > best_f1:
                best_f1 = f1
                best_alpha = alpha

        logger.info(f"[Validation] Best α = {best_alpha} (F1 = {best_f1:.4f})")
        return best_alpha, {"table": results_table, "best_f1": best_f1}

    def grid_search_beta(
        self,
        scored_segments: List[Segment],
        bank_id: str,
        year: int,
        agg_cfg: AggregationConfig,
        n_samples: int = 50,
    ) -> Tuple[Tuple[float, float, float], Dict]:
        """
        Random search over (β1, β2, β3) simplex to maximize F1.
        β1 + β2 + β3 = 1.
        """
        rng = np.random.default_rng(42)
        best_betas = (agg_cfg.beta1, agg_cfg.beta2, agg_cfg.beta3)
        best_f1 = 0.0

        for _ in range(n_samples):
            # Sample from Dirichlet (uniform on simplex)
            betas = rng.dirichlet([1, 1, 1])
            b1, b2, b3 = float(betas[0]), float(betas[1]), float(betas[2])

            for seg in scored_segments:
                seg.S_j = b1 * seg.S1_j + b2 * seg.S2_j + b3 * seg.P_llm_j

            metrics = self.evaluate_pipeline(scored_segments, bank_id, year)
            f1 = metrics.get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_betas = (round(b1, 3), round(b2, 3), round(b3, 3))

        logger.info(
            f"[Validation] Best β = {best_betas} (F1 = {best_f1:.4f})"
        )
        return best_betas, {"best_f1": best_f1}
