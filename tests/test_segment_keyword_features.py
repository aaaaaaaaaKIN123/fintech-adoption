from fintech_adoption.segment_keyword_features import compute_segment_keyword_features
from fintech_adoption.preprocessor import Segment


def test_segment_keyword_features_detect_adoption_context():
    segment = Segment(
        seg_id=1,
        text="Ngân hàng đã triển khai eKYC trên ứng dụng mobile banking cho khách hàng cá nhân.",
        page_num=3,
        char_offset=0,
    )

    features = compute_segment_keyword_features(segment)

    assert features["keyword_total_hits"] > 0
    assert features["matched_concept_count"] >= 1
    assert "ekyc_identity" in features["matched_concept_ids"]
    assert features["has_action_verb"] is True
    assert features["has_bank_self_action"] is True
    assert features["has_specific_product_or_process"] is True


def test_segment_keyword_features_flag_future_plan_marker():
    segment = Segment(
        seg_id=2,
        text="Ngân hàng dự kiến triển khai open banking trong năm tới.",
        page_num=4,
        char_offset=0,
    )

    features = compute_segment_keyword_features(segment)

    assert features["has_future_plan_marker"] is True
    assert features["future_plan_marker_count"] > 0
