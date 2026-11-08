import pandas as pd
import pytest

from fintech_adoption.stacked_adoption_model import (
    aggregate_stacked_panel,
    predict_segment_probabilities,
    train_stacked_logit,
    normalize_gold_labels,
)


@pytest.mark.parametrize("label", ["", " ", None, "ADOPETD"])
def test_incomplete_or_unknown_labels_are_rejected(label):
    labels = pd.DataFrame([{"bank_id": "A", "year": 2023, "seg_id": 1, "label": label}])
    with pytest.raises(ValueError, match="blank or unknown"):
        normalize_gold_labels(labels)


@pytest.mark.parametrize("value", [None, 0.5, 2])
def test_non_binary_gold_values_are_rejected(value):
    labels = pd.DataFrame([{"bank_id": "A", "year": 2023, "seg_id": 1, "y_adopted": value}])
    with pytest.raises(ValueError, match="0 or 1"):
        normalize_gold_labels(labels)


def test_duplicate_gold_keys_are_rejected():
    row = {"bank_id": "A", "year": 2023, "seg_id": 1, "label": "ADOPTED"}
    with pytest.raises(ValueError, match="one row per"):
        normalize_gold_labels(pd.DataFrame([row, row]))


def _feature_row(bank_id, year, seg_id, adopted_signal):
    return {
        "bank_id": bank_id,
        "year": year,
        "seg_id": seg_id,
        "canonical_clean_words": 10_000,
        "S1_j": 0.8 if adopted_signal else 0.1,
        "S2_j": 0.7 if adopted_signal else 0.1,
        "P_llm_j": 0.9 if adopted_signal else 0.0,
        "llm_confidence_raw": 0.9 if adopted_signal else 0.6,
        "llm_adoption_probability": 0.9 if adopted_signal else 0.0,
        "llm_bank_is_actor": int(adopted_signal),
        "llm_specific_capability": int(adopted_signal),
        "llm_future_or_plan_only": int(not adopted_signal),
        "keyword_total_hits": 3 if adopted_signal else 0,
        "matched_concept_count": 2 if adopted_signal else 0,
        "adoption_verb_count": 1 if adopted_signal else 0,
        "future_plan_marker_count": 0 if adopted_signal else 1,
        "has_action_verb": int(adopted_signal),
        "has_future_plan_marker": int(not adopted_signal),
        "has_bank_self_action": int(adopted_signal),
        "has_specific_product_or_process": int(adopted_signal),
        "keyword_density": 0.08 if adopted_signal else 0.0,
    }


def test_stacked_logit_trains_predicts_and_aggregates():
    features = pd.DataFrame(
        [
            _feature_row("A", 2020, 1, True),
            _feature_row("A", 2020, 2, False),
            _feature_row("B", 2020, 1, True),
            _feature_row("B", 2020, 2, False),
            _feature_row("C", 2021, 1, True),
            _feature_row("C", 2021, 2, False),
        ]
    )
    labels = pd.DataFrame(
        [
            {"bank_id": row.bank_id, "year": row.year, "seg_id": row.seg_id, "label": "ADOPTED" if idx % 2 == 0 else "IRRELEVANT"}
            for idx, row in enumerate(features.itertuples(index=False))
        ]
    )

    model, metrics = train_stacked_logit(features, labels)
    scored = predict_segment_probabilities(features, model)
    panel = aggregate_stacked_panel(scored, threshold=0.5)

    assert metrics["n_labeled_segments"] == 6
    assert "stacked_adopt_prob" in scored.columns
    assert scored["stacked_adopt_prob"].between(0, 1).all()
    assert {"stacked_FTA_it", "stacked_AdoptDummy_it", "stacked_FTA_HC_it"}.issubset(panel.columns)
    assert len(panel) == 3
