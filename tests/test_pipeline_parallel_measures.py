from pathlib import Path

from fintech_adoption.aggregator import BankYearRecord
from fintech_adoption.config import PipelineConfig
from fintech_adoption.pipeline import build_and_save_panel_outputs
from fintech_adoption.preprocessor import BankReport, Segment


def test_pipeline_output_builder_saves_hybrid_keyword_entropy_measures(tmp_path):
    cfg = PipelineConfig()
    cfg.output_dir = str(tmp_path)
    cfg.log_dir = str(tmp_path / "logs")
    records = [BankYearRecord(bank_id="VCB", year=2023, FTA_it=0.72)]
    reports = [
        BankReport(
            bank_id="VCB",
            year=2023,
            pdf_path="fake.pdf",
            segments=[
                Segment(
                    seg_id=0,
                    text="The bank launched eKYC and mobile banking.",
                    page_num=1,
                    char_offset=0,
                )
            ],
            canonical_clean_words=10_000,
            metadata={
                "pdf_backend": "ocrmypdf+pymupdf4llm",
                "extraction_status": "OK",
                "ocr_used": True,
                "scan_detected": True,
                "text_coverage": 0.75,
            },
        )
    ]

    panel = build_and_save_panel_outputs(records, reports, cfg)

    assert {"hybrid_FTA_it", "keyword_FTI", "entropy_FTI"}.issubset(panel.columns)
    assert panel.loc[0, "hybrid_FTA_it"] == 0.72
    assert panel.loc[0, "keyword_FTI"] > 0
    assert Path(tmp_path, "FTA_panel.csv").exists()
    assert Path(tmp_path, "FTA_baseline_indices.csv").exists()
    assert Path(cfg.log_dir, "pdf_extraction_quality.csv").exists()
