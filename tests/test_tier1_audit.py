import numpy as np

from fintech_adoption.config import Tier1Config
from fintech_adoption.preprocessor import BankReport, Segment
from fintech_adoption.tier1_scorer import score_tier1


class FakeEmbedder:
    seed_embeddings = np.eye(3, dtype=np.float32)

    def seed_relevance_scores(self, seg_embeddings, use_max=False, top_r=2):
        return np.array([0.9, 0.2], dtype=np.float32)

    def seed_relevance_details(self, seg_embeddings, top_n=3):
        return [
            [("ekyc_identity", 0.9), ("mobile_banking", 0.3), ("cloud_computing", 0.1)],
            [("cloud_computing", 0.2), ("mobile_banking", 0.1), ("ekyc_identity", 0.0)],
        ]

    @staticmethod
    def coherence_scores(candidate_embeddings):
        return np.ones(candidate_embeddings.shape[0], dtype=np.float32)


def test_tier1_saves_nearest_seed_audit_fields():
    report = BankReport(
        bank_id="VCB",
        year=2023,
        pdf_path="fake.pdf",
        segments=[
            Segment(seg_id=0, text="deployed ekyc for onboarding", page_num=1, char_offset=0),
            Segment(seg_id=1, text="generic business update", page_num=1, char_offset=50),
        ],
        canonical_clean_words=100,
    )
    seg_embeddings = np.eye(2, 3, dtype=np.float32)

    top_segments, _ = score_tier1(report, seg_embeddings, FakeEmbedder(), Tier1Config(top_k=1))

    seg = top_segments[0]
    assert seg.nearest_seed_ids == ["ekyc_identity", "mobile_banking", "cloud_computing"]
    assert seg.nearest_seed_scores[0] == 0.9
