from fintech_adoption.aggregator import aggregate_report
from fintech_adoption.config import AggregationConfig
from fintech_adoption.preprocessor import (
    BankReport,
    Segment,
    deduplicate_bilingual_segments,
    detect_language,
)


def test_bank_report_tracks_clean_canonical_and_retained_words_separately():
    segments = [
        Segment(seg_id=0, text="digital banking adoption deployed successfully", page_num=1, char_offset=0),
        Segment(seg_id=1, text="mobile banking launched with ekyc onboarding", page_num=1, char_offset=50),
    ]

    report = BankReport(
        bank_id="VCB",
        year=2023,
        pdf_path="fake.pdf",
        segments=segments,
        clean_total_words=1000,
        raw_total_words=1200,
        canonical_clean_words=900,
    )

    assert report.clean_total_words == 1000
    assert report.raw_total_words == 1200
    assert report.canonical_clean_words == 900
    assert report.retained_segment_words == 11
    assert report.total_words == 900


def test_aggregation_uses_canonical_clean_words_denominator():
    seg = Segment(seg_id=0, text="bank deployed eKYC", page_num=1, char_offset=0)
    seg.S1_j = 1.0
    seg.S2_j = 1.0
    seg.P_llm_j = 1.0
    seg.llm_label = "ADOPTED"

    report = BankReport(
        bank_id="VCB",
        year=2023,
        pdf_path="fake.pdf",
        segments=[seg],
        clean_total_words=40_000,
        canonical_clean_words=20_000,
    )

    record = aggregate_report(report, [seg], AggregationConfig(), run_llm=True)

    assert record.FTA_it == 0.5
    assert record.total_words == 20_000
    assert record.clean_total_words == 40_000
    assert record.canonical_clean_words == 20_000


def test_detect_language_marks_vietnamese_english_and_mixed_segments():
    assert detect_language("Ngân hàng đã triển khai định danh điện tử cho khách hàng.") == "vi"
    assert detect_language("The bank launched mobile banking for customers.") == "en"
    assert detect_language("Ngân hàng launched eKYC for khách hàng.") == "mixed"


def test_bilingual_dedup_keeps_one_canonical_segment_and_records_audit():
    segments = [
        Segment(
            seg_id=0,
            text="Ngân hàng đã triển khai eKYC cho khách hàng cá nhân trong năm 2023.",
            page_num=1,
            char_offset=0,
        ),
        Segment(
            seg_id=1,
            text="The bank implemented eKYC for individual customers in 2023.",
            page_num=2,
            char_offset=120,
        ),
        Segment(
            seg_id=2,
            text="Tổng tài sản tăng trưởng ổn định trong năm 2023.",
            page_num=3,
            char_offset=240,
        ),
    ]

    canonical, duplicates = deduplicate_bilingual_segments(segments)

    assert [seg.seg_id for seg in canonical] == [0, 2]
    assert canonical[0].duplicate_seg_ids == [1]
    assert canonical[0].languages_in_cluster == ["vi", "en"]
    assert duplicates[0]["duplicate_seg_id"] == 1
    assert duplicates[0]["canonical_seg_id"] == 0
