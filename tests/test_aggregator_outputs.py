import json

from fintech_adoption.aggregator import aggregate_report, save_segment_detail
from fintech_adoption.config import AggregationConfig
from fintech_adoption.preprocessor import BankReport, Segment


def make_scored_segment():
    seg = Segment(seg_id=0, text="Bank launched eKYC.", page_num=3, char_offset=100)
    seg.R_j = 0.8
    seg.C_j = 0.6
    seg.S1_j = 0.73
    seg.S2_j = 0.50
    seg.llm_label = "ADOPTED"
    seg.llm_confidence_raw = 0.90
    seg.P_llm_j = 0.90
    seg.llm_fintech_type = "eKYC"
    seg.llm_evidence = "The bank launched eKYC."
    seg.nearest_seed_ids = ["ekyc_identity"]
    seg.nearest_seed_scores = [0.81]
    seg.key_phrases = ["launched eKYC"]
    seg.language = "en"
    return seg


def test_segment_detail_contains_audit_fields(tmp_path):
    seg = make_scored_segment()

    out = save_segment_detail("VCB", 2023, [seg], str(tmp_path))
    row = json.loads(open(out, encoding="utf-8").readline())

    assert row["nearest_seed_ids"] == ["ekyc_identity"]
    assert row["llm_fintech_type"] == "eKYC"
    assert row["llm_evidence"] == "The bank launched eKYC."
    assert row["language"] == "en"
    assert row["text"] == "Bank launched eKYC."


def test_record_contains_denominator_and_candidate_counts():
    seg = make_scored_segment()
    report = BankReport(
        "VCB",
        2023,
        "fake.pdf",
        [seg],
        clean_total_words=10_000,
        raw_total_words=12_000,
        canonical_clean_words=9_000,
    )

    record = aggregate_report(report, [seg], AggregationConfig(), run_llm=True)

    assert record.FTA_it > 0
    assert record.clean_total_words == 10_000
    assert record.raw_total_words == 12_000
    assert record.canonical_clean_words == 9_000
    assert record.retained_segment_words == seg.word_count
