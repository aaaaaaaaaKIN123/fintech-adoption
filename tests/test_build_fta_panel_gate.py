import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "scripts"
    / "01_build_fta_panel.py"
)
SPEC = importlib.util.spec_from_file_location("build_fta_panel", SCRIPT_PATH)
build_fta_panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_fta_panel)

def test_enforce_extraction_quality_gate_blocks_inventory_rows_with_review_issues(tmp_path):
    quality_path = tmp_path / "pdf_extraction_quality.csv"
    pd.DataFrame(
        [
            {
                "bank_id": "VCB",
                "year": 2024,
                "pdf_path": "data/raw/vcb.pdf",
                "extraction_status": "OK",
                "scan_detected": False,
                "text_coverage": 1.0,
                "garbled_ratio": 0.0,
                "extraction_warnings": "",
                "read_error": "",
            },
            {
                "bank_id": "ACB",
                "year": 2020,
                "pdf_path": "data/raw/acb.pdf",
                "extraction_status": "NEED_MANUAL_REVIEW",
                "scan_detected": True,
                "text_coverage": 0.0,
                "garbled_ratio": 0.0,
                "extraction_warnings": "NEEDS_OCR",
                "read_error": "",
            },
        ]
    ).to_csv(quality_path, index=False)

    file_map = {
        ("VCB", 2024): "data/raw/vcb.pdf",
        ("ACB", 2020): "data/raw/acb.pdf",
    }

    with pytest.raises(build_fta_panel.ExtractionQualityGateError) as excinfo:
        build_fta_panel.enforce_extraction_quality_gate(file_map, str(quality_path))

    assert "ACB 2020" in str(excinfo.value)
    assert "NEED_MANUAL_REVIEW" in str(excinfo.value)

def test_enforce_extraction_quality_gate_can_allow_manual_review_rows(tmp_path):
    quality_path = tmp_path / "pdf_extraction_quality.csv"
    pd.DataFrame(
        [
            {
                "bank_id": "ACB",
                "year": 2020,
                "pdf_path": "data/raw/acb.pdf",
                "extraction_status": "NEED_MANUAL_REVIEW",
                "scan_detected": True,
                "text_coverage": 0.0,
                "garbled_ratio": 0.0,
                "extraction_warnings": "NEEDS_OCR",
                "read_error": "",
            }
        ]
    ).to_csv(quality_path, index=False)

    file_map = {("ACB", 2020): "data/raw/acb.pdf"}

    build_fta_panel.enforce_extraction_quality_gate(
        file_map,
        str(quality_path),
        allow_manual_review=True,
    )
