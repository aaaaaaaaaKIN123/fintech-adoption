"""
config.py — Centralized hyperparameters and settings for the
FinTech Adoption (FTA) measurement pipeline.

All tunable values live here so that calibration/validation
can be done by changing this file alone.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List


def resolve_project_root() -> Path:
    """Return the research workspace root."""
    return Path(os.environ.get("FTA_WORKSPACE", Path.cwd())).expanduser().resolve()


@dataclass
class EmbeddingConfig:
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    batch_size: int = 64
    device: str = "cpu"
    cache_dir: str = field(
        default_factory=lambda: str(
            resolve_project_root() / "results" / "models" / "embeddings"
        )
    )


@dataclass
class SegmentationConfig:
    min_segment_words: int = 15
    max_segment_words: int = 200
    window_overlap: int = 1
    bilingual_dedupe_threshold: float = 0.55


@dataclass
class ExtractionConfig:
    preferred_backend: str = "pymupdf4llm"
    fallback_backend: str = "pymupdf_fast"
    backend_timeout_seconds: int = 180
    enable_ocr: bool = True
    ocr_mode: str = "skip"
    ocr_output_dir: str = field(
        default_factory=lambda: str(resolve_project_root() / "data" / "processed" / "ocr")
    )
    ocr_invalidate_digital_signatures: bool = False
    ocr_jobs: int = 2
    ocr_timeout_seconds: int = 1800
    min_words_per_text_page: int = 10
    min_text_page_share: float = 0.50
    long_document_min_pages: int = 30
    min_words_per_page_long_document: float = 80.0


@dataclass
class Tier1Config:
    top_r_seeds: int = 5
    use_max: bool = False
    top_k: int = 30
    alpha: float = 0.65
    use_graph_centrality: bool = False
    pagerank_damping: float = 0.85


@dataclass
class Tier2Config:
    ngram_min: int = 1
    ngram_max: int = 4
    valid_pos_patterns: List[str] = field(default_factory=lambda: [
        "NOUN", "PROPN",
        "ADJ NOUN", "ADJ PROPN",
        "NOUN NOUN", "PROPN PROPN",
        "VERB NOUN", "VERB PROPN",
        "ADJ NOUN NOUN", "NOUN NOUN NOUN",
    ])
    adoption_verbs_vi: List[str] = field(default_factory=lambda: [
        "triển khai", "ứng dụng", "tích hợp", "đầu tư", "ra mắt",
        "áp dụng", "phát triển", "nâng cấp", "mở rộng", "số hóa",
        "vận hành", "khai thác", "sử dụng", "chuyển đổi", "cung cấp",
    ])
    adoption_verbs_en: List[str] = field(default_factory=lambda: [
        "deploy", "implement", "integrate", "invest", "launch",
        "adopt", "develop", "upgrade", "expand", "digitize",
        "operate", "leverage", "utilize", "transform", "offer",
        "provide", "automate", "pilot", "roll out",
    ])
    lambda1: float = 0.40
    lambda2: float = 0.30
    lambda3: float = 0.30
    top_phrases_per_para: int = 3


@dataclass
class LLMAdjudicationConfig:
    model: str = "gpt-5.5"
    provider: str = "openai"
    max_tokens: int = 512
    temperature: float = 0.0
    label_adopted: str = "ADOPTED"
    label_discussed: str = "DISCUSSED_ONLY"
    label_irrelevant: str = "IRRELEVANT"
    adoption_probability_by_label: dict = field(default_factory=lambda: {
        "ADOPTED": 1.0,
        "DISCUSSED_ONLY": 0.0,
        "IRRELEVANT": 0.0,
    })


@dataclass
class AggregationConfig:
    beta1: float = 0.30
    beta2: float = 0.30
    beta3: float = 0.40
    normalization_unit: int = 10_000
    min_score_threshold: float = 0.05


@dataclass
class PipelineConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    tier1: Tier1Config = field(default_factory=Tier1Config)
    tier2: Tier2Config = field(default_factory=Tier2Config)
    llm: LLMAdjudicationConfig = field(default_factory=LLMAdjudicationConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)

    log_dir: str = field(
        default_factory=lambda: str(resolve_project_root() / "results" / "logs")
    )
    output_dir: str = field(
        default_factory=lambda: str(resolve_project_root() / "data" / "analysis_output")
    )
    save_intermediate: bool = True
    run_llm_adjudication: bool = True
    allow_partial_reports: bool = False
