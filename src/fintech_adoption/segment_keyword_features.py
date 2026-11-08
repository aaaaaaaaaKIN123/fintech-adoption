"""
Segment-level keyword and adoption-context features.

These features form the transparent keyword branch for supervised stacking.
They are intentionally lightweight heuristics over already-filtered annual
report segments, not a replacement for semantic retrieval or LLM adjudication.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List

from .config import Tier2Config
from .preprocessor import Segment
from .seed_concepts import ALL_SEED_CONCEPTS

FUTURE_PLAN_MARKERS = [
    "sẽ",
    "dự kiến",
    "kế hoạch",
    "sắp triển khai",
    "định hướng",
    "mục tiêu",
    "will",
    "plan",
    "plans",
    "planned",
    "intend",
    "expected to",
    "aim",
    "aims",
]

BANK_ACTOR_MARKERS = [
    "ngân hàng",
    "nhà băng",
    "chúng tôi",
    "bank",
    "the bank",
    "we",
    "our",
]

SPECIFIC_CAPABILITY_MARKERS = [
    "ứng dụng",
    "dịch vụ",
    "sản phẩm",
    "quy trình",
    "hệ thống",
    "nền tảng",
    "giải pháp",
    "app",
    "service",
    "product",
    "process",
    "system",
    "platform",
    "solution",
]


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _phrase_count(text_norm: str, phrase: str) -> int:
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return 0
    pattern = r"(?<![a-z0-9])" + re.escape(phrase_norm) + r"(?![a-z0-9])"
    return len(re.findall(pattern, text_norm))


def _count_any(text_norm: str, phrases: Iterable[str]) -> int:
    return sum(_phrase_count(text_norm, phrase) for phrase in phrases)


def _has_actor_near_action(text_norm: str, action_phrases: List[str], window_tokens: int = 18) -> bool:
    tokens = text_norm.split()
    if not tokens:
        return False

    actor_positions = [
        idx
        for idx, token in enumerate(tokens)
        if token in {"bank", "we", "our"} or token.startswith("ngan")
    ]
    if not actor_positions:
        return False

    action_tokens = {normalize_text(phrase).split()[0] for phrase in action_phrases if normalize_text(phrase)}
    action_positions = [idx for idx, token in enumerate(tokens) if token in action_tokens]
    return any(
        abs(actor_idx - action_idx) <= window_tokens
        for actor_idx in actor_positions
        for action_idx in action_positions
    )


def compute_segment_keyword_features(
    segment: Segment,
    tier2_cfg: Tier2Config | None = None,
) -> Dict[str, object]:
    """Compute segment-level keyword/adoption-context features."""
    cfg = tier2_cfg or Tier2Config()
    text_norm = normalize_text(segment.text)
    word_count = max(segment.word_count, 1)

    matched_concepts: List[str] = []
    total_hits = 0
    for concept in ALL_SEED_CONCEPTS:
        hits = sum(_phrase_count(text_norm, alias) for alias in concept.aliases)
        if hits:
            matched_concepts.append(concept.concept_id)
            total_hits += hits

    adoption_verbs = cfg.adoption_verbs_vi + cfg.adoption_verbs_en
    adoption_verb_count = _count_any(text_norm, adoption_verbs)
    future_plan_count = _count_any(text_norm, FUTURE_PLAN_MARKERS)
    bank_actor_count = _count_any(text_norm, BANK_ACTOR_MARKERS)
    capability_marker_count = _count_any(text_norm, SPECIFIC_CAPABILITY_MARKERS)

    has_action_verb = adoption_verb_count > 0
    has_future_plan_marker = future_plan_count > 0
    has_bank_self_action = bank_actor_count > 0 and (
        has_action_verb or _has_actor_near_action(text_norm, adoption_verbs)
    )
    has_specific_product_or_process = bool(matched_concepts) and capability_marker_count > 0

    return {
        "keyword_total_hits": int(total_hits),
        "matched_concept_count": int(len(matched_concepts)),
        "matched_concept_ids": matched_concepts,
        "adoption_verb_count": int(adoption_verb_count),
        "future_plan_marker_count": int(future_plan_count),
        "has_action_verb": bool(has_action_verb),
        "has_future_plan_marker": bool(has_future_plan_marker),
        "has_bank_self_action": bool(has_bank_self_action),
        "has_specific_product_or_process": bool(has_specific_product_or_process),
        "keyword_density": round(float(total_hits) / word_count, 8),
    }


def annotate_segments_keyword_features(
    segments: Iterable[Segment],
    tier2_cfg: Tier2Config | None = None,
) -> None:
    """Attach keyword/adoption-context features to Segment.keyword_features."""
    for segment in segments:
        segment.keyword_features = compute_segment_keyword_features(segment, tier2_cfg)
