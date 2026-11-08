from pathlib import Path

import pandas as pd

from fintech_adoption.preprocessor import BankReport, Segment
from fintech_adoption.segment_feature_table import build_segment_feature_table, save_segment_feature_table


def test_segment_feature_table_saves_scored_candidate_rows(tmp_path):
    segment = Segment(
        seg_id=7,
        text="The bank launched mobile banking.",
        page_num=2,
        char_offset=10,
    )
    segment.S1_j = 0.5
    segment.S2_j = 0.4
    segment.P_llm_j = 0.8
    segment.S_j = 0.6
    segment.llm_label = "ADOPTED"
    segment.keyword_features = {
        "keyword_total_hits": 2,
        "matched_concept_count": 1,
        "matched_concept_ids": ["mobile_banking"],
        "has_action_verb": True,
    }
    report = BankReport(
        bank_id="VCB",
        year=2022,
        pdf_path="fake.pdf",
        segments=[segment],
        canonical_clean_words=12_000,
    )

    table = build_segment_feature_table([report])
    output_path = save_segment_feature_table([report], str(tmp_path))

    assert len(table) == 1
    assert table.loc[0, "bank_id"] == "VCB"
    assert table.loc[0, "matched_concept_ids"] == "mobile_banking"
    assert Path(output_path).exists()
    saved = pd.read_csv(output_path)
    assert saved.loc[0, "canonical_clean_words"] == 12_000
