"""
pipeline.py — End-to-end FinTech Adoption (FTA) measurement pipeline.

Full hybrid-stacking pipeline:
  Phase 0 → Seed concept set construction
  Phase 1 → PDF preprocessing & segmentation
  Phase 2 → Multilingual embedding computation
  Phase 3 → Tier-1 paragraph-level scoring (R_j, C_j, S_j^(1))
  Phase 4 → Tier-2 phrase-level scoring   (S_jl^(2))
  Phase 5 → LLM adjudication             (P_j^LLM, label)
  Phase 6 → Final score & FTA aggregation (FTA_it)

Usage
-----
  # Single report
  python pipeline.py \
    --pdf data/VCB_2022.pdf \
    --bank_id VCB --year 2022 \
    --mock_llm

  # Full batch
  python pipeline.py --config my_config.json --batch data/report_list.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Pipeline modules
from .config import PipelineConfig
from .seed_concepts import ALL_SEED_CONCEPTS, get_concept_texts, get_concept_ids
from .preprocessor import preprocess_report, preprocess_batch, BankReport
from .embedder import FinTechEmbedder
from .tier1_scorer import score_tier1
from .tier2_scorer import Tier2Scorer, CorpusTFIDF
from .llm_adjudicator import LLMAdjudicator
from .segment_keyword_features import annotate_segments_keyword_features
from .segment_feature_table import save_segment_feature_table
from .aggregator import (
    aggregate_report, build_panel,
    save_panel_csv, save_records_json, save_segment_detail,
    BankYearRecord
)
from .baseline_indices import build_baseline_panel, merge_hybrid_and_baselines

# Logging setup
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("pipeline")


class BatchCompletenessError(RuntimeError):
    """Raised when a strict batch run cannot produce every selected bank-year."""


def save_failed_reports_log(failures: List[dict], log_dir: str) -> str:
    """Persist failed/missing bank-year rows for strict batch handoff."""
    if not failures:
        return ""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("[Batch] pandas not installed; skipped failed report log.")
        return ""

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "fta_failed_reports.csv")
    pd.DataFrame(failures).to_csv(path, index=False, encoding="utf-8-sig")
    logger.error("[Batch] Failed report log saved -> %s", path)
    return path


def enforce_batch_completeness(
    file_map: Dict[Tuple, str],
    reports: List[BankReport],
    failures: List[dict],
    cfg: PipelineConfig,
) -> None:
    """Fail strict runs when any selected bank-year is missing after preprocessing."""
    expected = {(str(bank_id), int(year)) for bank_id, year in file_map}
    observed = {(str(report.bank_id), int(report.year)) for report in reports}
    missing = sorted(expected - observed)
    rows = list(failures)
    for bank_id, year in missing:
        rows.append({
            "bank_id": bank_id,
            "year": year,
            "pdf_path": file_map.get((bank_id, year), ""),
            "stage": "preprocess",
            "error": "Selected bank-year did not produce a BankReport.",
        })
    if not rows:
        return

    log_path = save_failed_reports_log(rows, cfg.log_dir)
    examples = ", ".join(f"{bank_id} {year}" for bank_id, year in missing[:10])
    if not examples and rows:
        examples = ", ".join(f"{row.get('bank_id')} {row.get('year')}" for row in rows[:10])
    suffix = f" See {log_path}." if log_path else ""
    raise BatchCompletenessError(
        "Batch preprocessing did not produce every selected bank-year: "
        f"{examples}.{suffix}"
    )



# PHASE 0: SEED CONCEPT INITIALISATION


def phase0_init_seeds(cfg: PipelineConfig, embedder: FinTechEmbedder) -> None:
    """
    Initialise seed concept embeddings.
    Loads from cache if available, otherwise encodes and saves.
    """
    seed_cache = os.path.join(cfg.embedding.cache_dir, "seed_embeddings.pkl")

    if os.path.exists(seed_cache):
        logger.info("[Phase 0] Loading cached seed embeddings ...")
        embedder.load_seeds(seed_cache)
    else:
        logger.info("[Phase 0] Encoding seed concepts ...")
        seed_texts = get_concept_texts()
        embedder.encode_seeds(seed_texts)
        os.makedirs(cfg.embedding.cache_dir, exist_ok=True)
        embedder.save_seeds(seed_cache)

    logger.info(
        f"[Phase 0] Seed set ready: M={len(get_concept_ids())} concepts, "
        f"{len([a for c in ALL_SEED_CONCEPTS for a in c.aliases])} total aliases"
    )



# PHASE 1: PREPROCESSING


def phase1_preprocess(
    pdf_path: str,
    bank_id: str,
    year: int,
    cfg: PipelineConfig,
) -> BankReport:
    """Preprocess a single PDF into segmented BankReport."""
    logger.info(f"\n{'='*60}")
    logger.info(f"[Phase 1] Preprocessing: {bank_id} {year}")
    logger.info(f"{'='*60}")
    return preprocess_report(
        pdf_path=pdf_path,
        bank_id=bank_id,
        year=year,
        min_words=cfg.segmentation.min_segment_words,
        max_words=cfg.segmentation.max_segment_words,
        bilingual_dedupe_threshold=cfg.segmentation.bilingual_dedupe_threshold,
        fallback_backend=cfg.extraction.fallback_backend,
        backend_timeout_seconds=cfg.extraction.backend_timeout_seconds,
        enable_ocr=cfg.extraction.enable_ocr,
        ocr_mode=cfg.extraction.ocr_mode,
        ocr_output_dir=cfg.extraction.ocr_output_dir,
        ocr_invalidate_digital_signatures=cfg.extraction.ocr_invalidate_digital_signatures,
        ocr_jobs=cfg.extraction.ocr_jobs,
        ocr_timeout_seconds=cfg.extraction.ocr_timeout_seconds,
        min_words_per_text_page=cfg.extraction.min_words_per_text_page,
        min_text_page_share=cfg.extraction.min_text_page_share,
        long_document_min_pages=cfg.extraction.long_document_min_pages,
        min_words_per_page_long_document=cfg.extraction.min_words_per_page_long_document,
    )



# PHASE 2: EMBEDDING


def phase2_embed(
    report: BankReport,
    embedder: FinTechEmbedder,
) -> np.ndarray:
    """Embed all segments of a report. Returns (N, d) matrix."""
    logger.info(f"[Phase 2] Embedding {len(report.segments)} segments ...")
    seg_texts = [s.text for s in report.segments]
    embs = embedder.encode_segments(seg_texts, show_progress=len(seg_texts) > 100)
    return embs



# PHASE 3: TIER-1 SCORING


def phase3_tier1(
    report: BankReport,
    seg_embeddings: np.ndarray,
    embedder: FinTechEmbedder,
    cfg: PipelineConfig,
) -> Tuple[List, np.ndarray]:
    """
    Run Tier-1 scoring.
    Returns (top_k_segments, top_k_embeddings).
    """
    logger.info(f"[Phase 3] Tier-1 scoring (alpha={cfg.tier1.alpha}, K={cfg.tier1.top_k}) ...")
    return score_tier1(report, seg_embeddings, embedder, cfg.tier1)



# PHASE 4: TIER-2 SCORING


def phase4_tier2(
    top_k_segments: List,
    embedder: FinTechEmbedder,
    cfg: PipelineConfig,
    tfidf: Optional[CorpusTFIDF] = None,
) -> None:
    """Run Tier-2 phrase scoring in-place on top_k_segments."""
    logger.info("[Phase 4] Tier-2 phrase scoring ...")
    scorer = Tier2Scorer(
        embedder=embedder,
        cfg=cfg.tier2,
        tfidf=tfidf,
        use_spacy=False,  # Set True if spaCy is installed
    )
    scorer.score_segments(
        top_k_segments,
        top_n=cfg.tier2.top_phrases_per_para,
    )

def phase4b_keyword_features(top_k_segments: List, cfg: PipelineConfig) -> None:
    """Attach transparent keyword/adoption-context features to candidates."""
    logger.info("[Phase 4b] Segment-level keyword/adoption features ...")
    annotate_segments_keyword_features(top_k_segments, cfg.tier2)



# PHASE 5: LLM ADJUDICATION


def phase5_llm(
    top_k_segments: List,
    cfg: PipelineConfig,
    api_key: Optional[str] = None,
    mock: bool = False,
) -> None:
    """Run LLM adjudication in-place."""
    if not cfg.run_llm_adjudication:
        logger.info("[Phase 5] LLM adjudication skipped (disabled in config).")
        for seg in top_k_segments:
            seg.llm_label = "UNKNOWN"
            seg.llm_evidence = "LLM adjudication disabled for this run."
            seg.P_llm_j = 0.0
        return

    logger.info(f"[Phase 5] LLM adjudication ({'mock' if mock else cfg.llm.provider}) ...")
    adjudicator = LLMAdjudicator(
        cfg=cfg.llm,
        api_key=api_key,
        mock=mock,
        rate_limit_delay=0.3 if mock else 0.8,
    )
    adjudicator.adjudicate_segments(top_k_segments)



# PHASE 6: AGGREGATION


def phase6_aggregate(
    report: BankReport,
    top_k_segments: List,
    cfg: PipelineConfig,
) -> BankYearRecord:
    """Compute FTA_it and produce BankYearRecord."""
    logger.info("[Phase 6] Aggregating -> FTA index ...")
    return aggregate_report(
        report=report,
        top_k_segments=top_k_segments,
        cfg=cfg.aggregation,
        run_llm=cfg.run_llm_adjudication,
    )

def build_and_save_panel_outputs(
    records: List[BankYearRecord],
    reports: List[BankReport],
    cfg: PipelineConfig,
):
    """Build and save hybrid, keyword, and entropy FTA measures side by side."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_records_json(records, cfg.output_dir)

    panel = build_panel(records)
    baseline_panel = build_baseline_panel(reports)
    baseline_path = os.path.join(cfg.output_dir, "FTA_baseline_indices.csv")
    baseline_panel.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    logger.info(f"[Batch] Baseline indices saved -> {baseline_path}")

    panel = merge_hybrid_and_baselines(panel, baseline_panel)
    save_panel_csv(panel, cfg.output_dir)
    segment_feature_path = save_segment_feature_table(reports, cfg.output_dir)
    logger.info(f"[Batch] Segment feature table saved -> {segment_feature_path}")
    save_extraction_quality_log(reports, cfg.log_dir)
    return panel

def save_extraction_quality_log(
    reports: List[BankReport],
    log_dir: str,
) -> str:
    """Save report-level PDF extraction quality metadata."""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("[Batch] pandas not installed; skipped extraction quality log.")
        return ""

    os.makedirs(log_dir, exist_ok=True)
    rows = []
    page_rows = []
    for report in reports:
        metadata = report.metadata or {}
        rows.append({
            "bank_id": report.bank_id,
            "year": report.year,
            "pdf_path": report.pdf_path,
            "pdf_backend": metadata.get("pdf_backend", ""),
            "extraction_status": metadata.get("extraction_status", ""),
            "ocr_used": metadata.get("ocr_used", False),
            "ocr_engine": metadata.get("ocr_engine", ""),
            "ocr_pdf_path": metadata.get("ocr_pdf_path", ""),
            "scan_detected": metadata.get("scan_detected", False),
            "num_pages": metadata.get("num_pages", 0),
            "text_coverage": metadata.get("text_coverage", ""),
            "text_pages_ge10": metadata.get("text_pages_ge10", 0),
            "text_pages_ge50": metadata.get("text_pages_ge50", 0),
            "text_page_share_ge10": metadata.get("text_page_share_ge10", ""),
            "words_per_physical_page": metadata.get("words_per_physical_page", ""),
            "median_words_per_page": metadata.get("median_words_per_page", ""),
            "garbled_ratio": metadata.get("garbled_ratio", ""),
            "raw_total_words": report.raw_total_words,
            "clean_total_words": report.clean_total_words,
            "canonical_clean_words": report.canonical_clean_words,
            "n_raw_segments": metadata.get("n_raw_segments", 0),
            "n_canonical_segments": metadata.get("n_canonical_segments", 0),
            "extraction_warnings": " | ".join(metadata.get("extraction_warnings", [])),
        })
        for page in metadata.get("page_quality", []):
            page_rows.append({
                "bank_id": report.bank_id,
                "year": report.year,
                "pdf_path": report.pdf_path,
                "pdf_backend": metadata.get("pdf_backend", ""),
                "ocr_used": metadata.get("ocr_used", False),
                **page,
            })
    path = os.path.join(log_dir, "pdf_extraction_quality.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    page_path = os.path.join(log_dir, "pdf_page_quality.csv")
    pd.DataFrame(page_rows).to_csv(page_path, index=False, encoding="utf-8-sig")
    logger.info(f"[Batch] PDF extraction quality log saved -> {path}")
    logger.info(f"[Batch] PDF page quality log saved -> {page_path}")
    return path



# SINGLE REPORT PIPELINE


def run_single_report(
    pdf_path: str,
    bank_id: str,
    year: int,
    cfg: PipelineConfig,
    embedder: FinTechEmbedder,
    tfidf: Optional[CorpusTFIDF] = None,
    api_key: Optional[str] = None,
    mock_llm: bool = False,
    save_detail: bool = True,
) -> BankYearRecord:
    """
    Full pipeline for a single annual report.

    Parameters
    ----------
    pdf_path  : Path to PDF.
    bank_id   : Bank identifier string.
    year      : Report year.
    cfg       : PipelineConfig.
    embedder  : FinTechEmbedder with seeds already loaded.
    tfidf     : Pre-fitted CorpusTFIDF (optional, used in Tier-2).
    api_key   : LLM API key (Anthropic or OpenAI).
    mock_llm  : Use mock LLM backend (no API key needed).
    save_detail: Save segment-level JSONL for this report.

    Returns
    -------
    BankYearRecord with FTA_it and all component scores.
    """
    t_start = time.time()

    # Phase 1
    report = phase1_preprocess(pdf_path, bank_id, year, cfg)
    if not report.segments:
        logger.warning(f"No segments extracted for {bank_id} {year}. Returning empty record.")
        return BankYearRecord(bank_id=bank_id, year=year)

    if tfidf is None:
        logger.info("[Single] Fitting report-level TF-IDF ...")
        tfidf = CorpusTFIDF(ngram_range=(cfg.tier2.ngram_min, cfg.tier2.ngram_max))
        for idx, seg in enumerate(report.segments):
            seg.tfidf_doc_index = idx
        tfidf.fit([seg.text for seg in report.segments])

    # Phase 2
    seg_embs = phase2_embed(report, embedder)

    # Phase 3
    top_k_segs, _top_k_embs = phase3_tier1(report, seg_embs, embedder, cfg)

    # Phase 4
    phase4_tier2(top_k_segs, embedder, cfg, tfidf)
    phase4b_keyword_features(top_k_segs, cfg)

    # Phase 5
    phase5_llm(top_k_segs, cfg, api_key, mock_llm)

    # Phase 6
    record = phase6_aggregate(report, top_k_segs, cfg)

    # Intermediate save
    if save_detail and cfg.save_intermediate:
        os.makedirs(cfg.output_dir, exist_ok=True)
        save_segment_detail(bank_id, year, top_k_segs, cfg.output_dir)

    elapsed = time.time() - t_start
    logger.info(
        f"[Pipeline] OK {bank_id} {year}: FTA={record.FTA_it:.4f} | "
        f"Adopted={record.n_segments_adopted} segs | "
        f"Time={elapsed:.1f}s"
    )
    return record



# BATCH PIPELINE


def run_batch(
    file_map: Dict[Tuple, str],   # {(bank_id, year): pdf_path}
    cfg: PipelineConfig,
    api_key: Optional[str] = None,
    mock_llm: bool = False,
    fit_tfidf: bool = True,
) -> List[BankYearRecord]:
    """
    Run the full pipeline on a batch of annual reports.

    Steps:
    1. Pre-process all reports.
    2. Fit corpus-level TF-IDF.
    3. Encode seeds.
    4. Run single-report pipeline for each.
    5. Assemble panel dataset.

    Parameters
    ----------
    file_map  : Dict mapping (bank_id, year) → pdf_path.
    cfg       : PipelineConfig.
    api_key   : LLM API key.
    mock_llm  : Use mock LLM.
    fit_tfidf : Pre-fit corpus TF-IDF across all reports.
    """
    logger.info(f"\n{'='*60}")
    logger.info("  FINTECH ADOPTION PIPELINE - BATCH MODE")
    logger.info(f"  Reports: {len(file_map)}")
    logger.info(f"{'='*60}\n")

    # Initialise embedder
    embedder = FinTechEmbedder(
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
        cache_dir=cfg.embedding.cache_dir,
        batch_size=cfg.embedding.batch_size,
    )
    phase0_init_seeds(cfg, embedder)

    # Pre-process all reports
    logger.info("[Batch] Phase 1: Pre-processing all reports ...")
    preprocess_failures: List[dict] = []
    reports = preprocess_batch(
        file_map={k: v for k, v in file_map.items()},
        min_words=cfg.segmentation.min_segment_words,
        max_words=cfg.segmentation.max_segment_words,
        bilingual_dedupe_threshold=cfg.segmentation.bilingual_dedupe_threshold,
        fallback_backend=cfg.extraction.fallback_backend,
        backend_timeout_seconds=cfg.extraction.backend_timeout_seconds,
        enable_ocr=cfg.extraction.enable_ocr,
        ocr_mode=cfg.extraction.ocr_mode,
        ocr_output_dir=cfg.extraction.ocr_output_dir,
        ocr_invalidate_digital_signatures=cfg.extraction.ocr_invalidate_digital_signatures,
        ocr_jobs=cfg.extraction.ocr_jobs,
        ocr_timeout_seconds=cfg.extraction.ocr_timeout_seconds,
        min_words_per_text_page=cfg.extraction.min_words_per_text_page,
        min_text_page_share=cfg.extraction.min_text_page_share,
        long_document_min_pages=cfg.extraction.long_document_min_pages,
        min_words_per_page_long_document=cfg.extraction.min_words_per_page_long_document,
        failures=preprocess_failures,
    )
    if not getattr(cfg, "allow_partial_reports", False):
        enforce_batch_completeness(file_map, reports, preprocess_failures, cfg)
    elif preprocess_failures:
        save_failed_reports_log(preprocess_failures, cfg.log_dir)

    # Fit corpus TF-IDF
    tfidf = None
    if fit_tfidf:
        logger.info("[Batch] Fitting corpus TF-IDF ...")
        all_texts = []
        doc_index = 0
        for report in reports:
            for seg in report.segments:
                seg.tfidf_doc_index = doc_index
                all_texts.append(seg.text)
                doc_index += 1
        if all_texts:
            tfidf = CorpusTFIDF(ngram_range=(cfg.tier2.ngram_min, cfg.tier2.ngram_max))
            tfidf.fit(all_texts)

    # Per-report pipeline
    records: List[BankYearRecord] = []
    for (bank_id, year), pdf_path in file_map.items():
        # Find matching report
        matching = [r for r in reports if r.bank_id == bank_id and r.year == year]
        if not matching:
            logger.error(f"[Batch] No processed report found for {bank_id} {year}.")
            continue
        report = matching[0]

        # Embed
        seg_embs = phase2_embed(report, embedder)

        # Tier 1
        top_k_segs, _ = score_tier1(report, seg_embs, embedder, cfg.tier1)

        # Tier 2
        phase4_tier2(top_k_segs, embedder, cfg, tfidf)
        phase4b_keyword_features(top_k_segs, cfg)

        # LLM
        phase5_llm(top_k_segs, cfg, api_key, mock_llm)

        # Aggregate
        record = phase6_aggregate(report, top_k_segs, cfg)
        records.append(record)

        # Save segment detail
        if cfg.save_intermediate:
            save_segment_detail(bank_id, year, top_k_segs, cfg.output_dir)

    # Build and save panel
    if records:
        try:
            panel = build_and_save_panel_outputs(records, reports, cfg)
            logger.info(f"\n[Batch] Panel shape: {panel.shape}")
            preview_cols = [
                "bank_id", "year", "hybrid_FTA_it", "keyword_FTI",
                "entropy_FTI", "FTA_binary", "n_segments_adopted",
            ]
            logger.info(panel[preview_cols].to_string(index=False))
        except ImportError:
            logger.warning("[Batch] pandas not installed; skipped CSV export.")

    logger.info(f"\n[Batch] Complete. {len(records)}/{len(file_map)} reports processed.")
    return records



# CLI


def main():
    parser = argparse.ArgumentParser(
        description="FinTech Adoption Measurement Pipeline"
    )
    parser.add_argument("--pdf",      type=str, help="Path to single PDF report")
    parser.add_argument("--bank_id",  type=str, default="BANK", help="Bank identifier")
    parser.add_argument("--year",     type=int, default=2022,  help="Report year")
    parser.add_argument("--batch",    type=str, help="CSV file: bank_id,year,pdf_path")
    parser.add_argument("--mock_llm", action="store_true", help="Use mock LLM (no API key)")
    parser.add_argument("--no_llm",   action="store_true", help="Skip LLM adjudication")
    parser.add_argument("--api_key",  type=str, help="Anthropic/OpenAI API key")
    parser.add_argument("--llm_provider", type=str, default=None, help="Override LLM provider")
    parser.add_argument("--llm_model", type=str, default=None, help="Override LLM model")
    parser.add_argument("--output",   type=str, default=None, help="Output directory")
    parser.add_argument("--top_k",    type=int, default=30,   help="Tier-1 top-K")
    parser.add_argument("--alpha",    type=float, default=0.65, help="alpha weight for S^(1)")
    parser.add_argument(
        "--allow_partial_reports",
        action="store_true",
        help="Allow exploratory batch output when some selected reports fail preprocessing.",
    )

    args = parser.parse_args()

    # Build config
    cfg = PipelineConfig()
    if args.output:
        cfg.output_dir = args.output
    cfg.tier1.top_k = args.top_k
    cfg.tier1.alpha = args.alpha
    cfg.allow_partial_reports = args.allow_partial_reports
    if args.llm_provider:
        cfg.llm.provider = args.llm_provider
    if args.llm_model:
        cfg.llm.model = args.llm_model
    cfg.run_llm_adjudication = not args.no_llm

    if cfg.llm.provider == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    else:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    # Single report mode
    if args.pdf:
        embedder = FinTechEmbedder(
            model_name=cfg.embedding.model_name,
            device=cfg.embedding.device,
            cache_dir=cfg.embedding.cache_dir,
        )
        phase0_init_seeds(cfg, embedder)
        record = run_single_report(
            pdf_path=args.pdf,
            bank_id=args.bank_id,
            year=args.year,
            cfg=cfg,
            embedder=embedder,
            api_key=api_key,
            mock_llm=args.mock_llm,
        )
        print(f"\n{'='*50}")
        print(f"  RESULT: {record.bank_id} {record.year}")
        print(f"  FTA_it        = {record.FTA_it:.6f}")
        print(f"  FTA_binary    = {record.FTA_binary}")
        print(f"  Adopted segs  = {record.n_segments_adopted}")
        print(f"  Discussed     = {record.n_segments_discussed}")
        print(f"  Top phrases   : {' | '.join(record.top_key_phrases[:5])}")
        print(f"{'='*50}")

    # Batch mode
    elif args.batch:
        try:
            import pandas as pd
            df = pd.read_csv(args.batch)
            file_map = {
                (row["bank_id"], int(row["year"])): row["pdf_path"]
                for _, row in df.iterrows()
            }
        except ImportError:
            # Manual CSV parsing
            file_map = {}
            with open(args.batch) as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        file_map[(parts[0], int(parts[1]))] = parts[2]

        run_batch(file_map, cfg, api_key=api_key, mock_llm=args.mock_llm)
    else:
        parser.print_help()
        print("\n[ERROR] Provide --pdf or --batch.")
        sys.exit(1)


if __name__ == "__main__":
    main()
