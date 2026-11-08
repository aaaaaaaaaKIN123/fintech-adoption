from fintech_adoption.aggregator import aggregate_report, compute_segment_final_score
from fintech_adoption.config import AggregationConfig
from fintech_adoption.preprocessor import BankReport, Segment


def make_segment(seg_id, label, s1=0.5, s2=0.3, p_llm=0.0):
    seg = Segment(
        seg_id=seg_id,
        text="The bank discusses digital banking and data platform adoption.",
        page_num=1,
        char_offset=0,
    )
    seg.S1_j = s1
    seg.S2_j = s2
    seg.P_llm_j = p_llm
    seg.llm_label = label
    seg.key_phrases = ["digital banking", "data platform"]
    return seg


def make_report(segments, canonical_words=10_000):
    return BankReport(
        bank_id="TCB",
        year=2025,
        pdf_path="fake.pdf",
        segments=segments,
        canonical_clean_words=canonical_words,
        clean_total_words=canonical_words,
        raw_total_words=canonical_words,
        metadata={"num_pages": 1},
    )


def test_discussed_only_segment_has_zero_final_score_when_llm_runs():
    seg = make_segment("discussed", "DISCUSSED_ONLY", s1=0.6, s2=0.4, p_llm=0.0)

    score = compute_segment_final_score(seg, AggregationConfig(), run_llm=True)

    assert score == 0.0
    assert seg.S_j == 0.0


def test_adopted_segment_keeps_weighted_final_score_when_llm_runs():
    seg = make_segment("adopted", "ADOPTED", s1=0.6, s2=0.4, p_llm=1.0)

    score = compute_segment_final_score(seg, AggregationConfig(), run_llm=True)

    assert score == 0.7
    assert seg.S_j == 0.7


def test_report_with_only_discussed_segments_has_zero_fta_but_keeps_discussion_count():
    segments = [
        make_segment(1, "DISCUSSED_ONLY", s1=0.6, s2=0.4),
        make_segment(2, "DISCUSSED_ONLY", s1=0.5, s2=0.3),
    ]
    report = make_report(segments)

    record = aggregate_report(report, segments, AggregationConfig(), run_llm=True)

    assert record.FTA_it == 0.0
    assert record.FTA_binary == 0
    assert record.n_segments_adopted == 0
    assert record.n_segments_discussed == 2
    assert record.n_scored_segments == 0
    assert record.top_key_phrases == []


def test_mixed_report_counts_only_adopted_segments_in_fta():
    adopted = make_segment(1, "ADOPTED", s1=0.6, s2=0.4, p_llm=1.0)
    discussed = make_segment(2, "DISCUSSED_ONLY", s1=0.9, s2=0.8, p_llm=0.0)
    report = make_report([adopted, discussed])

    record = aggregate_report(report, [adopted, discussed], AggregationConfig(), run_llm=True)

    assert record.FTA_it == 0.7
    assert record.FTA_binary == 1
    assert record.n_segments_adopted == 1
    assert record.n_segments_discussed == 1
    assert record.n_scored_segments == 1
    assert record.top_key_phrases == ["digital banking", "data platform"]
