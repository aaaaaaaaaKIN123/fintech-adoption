import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "scripts"
    / "04_check_report_pdf_quality.py"
)
SPEC = importlib.util.spec_from_file_location("report_pdf_quality", SCRIPT_PATH)
report_pdf_quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_pdf_quality)


def test_expected_bank_year_summary_counts_present_and_missing_rows():
    sample = pd.DataFrame(
        [
            {"bank_id": "VCB", "include_in_main_sample": True},
            {"bank_id": "BID", "include_in_main_sample": True},
            {"bank_id": "SCB", "include_in_main_sample": False},
        ]
    )
    inventory = pd.DataFrame(
        [
            {"bank_id": "VCB", "year": 2024},
            {"bank_id": "BID", "year": 2025},
        ]
    )

    summary = report_pdf_quality.expected_bank_year_summary(
        sample,
        inventory,
        years=[2024, 2025],
    )

    assert summary["expected_bank_years"] == 4
    assert summary["present_bank_years"] == 2
    assert summary["missing_bank_years"] == 2
    assert summary["missing_rate"] == 0.5


def test_current_missing_bank_year_rows_lists_absent_inventory_pairs():
    sample = pd.DataFrame(
        [
            {"bank_id": "VCB", "bank_name": "Vietcombank", "common_name": "VCB", "include_in_main_sample": True},
            {"bank_id": "SCB", "bank_name": "SCB", "common_name": "SCB", "include_in_main_sample": False},
        ]
    )
    inventory = pd.DataFrame([{"bank_id": "VCB", "year": 2024}])

    missing = report_pdf_quality.current_missing_bank_year_rows(sample, inventory, years=[2024, 2025])

    assert len(missing) == 1
    assert missing.iloc[0]["bank_id"] == "VCB"
    assert missing.iloc[0]["year"] == 2025
    assert missing.iloc[0]["missing_reason"] == "not_in_current_inventory"

def test_extraction_review_rows_lists_blocking_pdf_quality_issues():
    quality = pd.DataFrame(
        [
            {
                "bank_id": "VCB",
                "year": 2024,
                "pdf_path": "data/raw/vcb.pdf",
                "file_exists": True,
                "checksum_ok": True,
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
                "file_exists": True,
                "checksum_ok": True,
                "extraction_status": "NEED_MANUAL_REVIEW",
                "scan_detected": True,
                "text_coverage": 0.0,
                "garbled_ratio": 0.0,
                "extraction_warnings": "NEEDS_OCR",
                "read_error": "",
            },
            {
                "bank_id": "BID",
                "year": 2021,
                "pdf_path": "data/raw/bid.pdf",
                "file_exists": False,
                "checksum_ok": False,
                "extraction_status": "FILE_MISSING",
                "scan_detected": False,
                "text_coverage": 0.0,
                "garbled_ratio": 0.0,
                "extraction_warnings": "",
                "read_error": "",
            },
        ]
    )

    review = report_pdf_quality.extraction_review_rows(quality)

    assert list(review["bank_id"]) == ["ACB", "BID"]
    assert list(review["review_reason"]) == ["needs_ocr", "file_missing"]
    assert "recommended_action" in review.columns

def test_build_summary_records_ocr_dependency_status(monkeypatch):
    quality = pd.DataFrame(
        [
            {
                "bank_id": "VCB",
                "year": 2024,
                "file_exists": True,
                "checksum_ok": True,
                "extraction_status": "OK",
                "scan_detected": False,
                "file_size_bytes": 1024,
                "pdf_backend": "pymupdf_fast",
            }
        ]
    )
    sample = pd.DataFrame([{"bank_id": "VCB", "include_in_main_sample": True}])
    inventory = pd.DataFrame([{"bank_id": "VCB", "year": 2024}])

    monkeypatch.setattr(report_pdf_quality.shutil, "which", lambda name: None)
    ocr_status = report_pdf_quality.ocr_dependency_status(enable_ocr=True)
    summary = report_pdf_quality.build_summary(
        quality,
        sample,
        inventory,
        pd.DataFrame(),
        years=[2024],
        ocr_status=ocr_status,
    )

    assert summary["ocr_requested"] is True
    assert summary["ocr_available"] is False
    assert summary["ocr_engine"] == "ocrmypdf"

def test_resolve_quality_backend_uses_ocr_capable_backend_when_ocr_requested_by_default():
    assert report_pdf_quality.resolve_quality_backend(None, enable_ocr=True) == "pymupdf4llm"
    assert report_pdf_quality.resolve_quality_backend(None, enable_ocr=False) == "pymupdf_fast"
    assert report_pdf_quality.resolve_quality_backend("pymupdf_fast", enable_ocr=True) == "pymupdf_fast"

def test_select_inventory_rows_for_quality_refresh_keeps_only_existing_review_issues():
    inventory = pd.DataFrame(
        [
            {"bank_id": "VCB", "year": 2024, "pdf_path": "data/raw/vcb.pdf"},
            {"bank_id": "ACB", "year": 2020, "pdf_path": "data/raw/acb.pdf"},
            {"bank_id": "BID", "year": 2021, "pdf_path": "data/raw/bid.pdf"},
        ]
    )
    quality = pd.DataFrame(
        [
            {"bank_id": "VCB", "year": 2024, "extraction_status": "OK", "file_exists": True, "extraction_warnings": ""},
            {"bank_id": "ACB", "year": 2020, "extraction_status": "NEED_MANUAL_REVIEW", "file_exists": True, "extraction_warnings": "NEEDS_OCR"},
            {"bank_id": "BID", "year": 2021, "extraction_status": "READ_ERROR", "file_exists": True, "extraction_warnings": ""},
        ]
    )

    selected = report_pdf_quality.select_inventory_rows_for_quality_refresh(inventory, quality)

    assert list(selected["bank_id"]) == ["ACB", "BID"]

def test_merge_quality_refresh_rows_replaces_review_rows_without_dropping_ok_rows():
    existing = pd.DataFrame(
        [
            {"bank_id": "VCB", "year": 2024, "extraction_status": "OK"},
            {"bank_id": "ACB", "year": 2020, "extraction_status": "NEED_MANUAL_REVIEW"},
        ]
    )
    refreshed = pd.DataFrame([{"bank_id": "ACB", "year": 2020, "extraction_status": "OK"}])

    merged = report_pdf_quality.merge_quality_refresh_rows(existing, refreshed)

    assert list(merged["bank_id"]) == ["VCB", "ACB"]
    assert list(merged["extraction_status"]) == ["OK", "OK"]
