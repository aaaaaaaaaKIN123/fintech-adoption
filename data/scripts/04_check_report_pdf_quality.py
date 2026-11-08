"""Check downloaded annual-report PDF integrity and extraction quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from fintech_adoption.document_reader import DocumentReaderConfig, PageText, assess_text_quality, read_pdf_auto  # noqa: E402


def _as_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def _quality_key(bank_id, year) -> tuple[str, int]:
    return str(bank_id).strip().upper(), int(year)


def ocr_dependency_status(enable_ocr: bool, engine: str = "ocrmypdf") -> dict:
    """Return OCR dependency state for run metadata and quality summaries."""
    engine_path = shutil.which(engine)
    return {
        "ocr_requested": bool(enable_ocr),
        "ocr_engine": engine,
        "ocr_available": bool(engine_path),
        "ocr_engine_path": engine_path or "",
    }

def resolve_quality_backend(requested_backend: str | None, enable_ocr: bool) -> str:
    """Choose the quality-check backend, keeping OCR-enabled runs OCR-capable.

    The fast PyMuPDF path is useful for a cheap text-layer audit, but it never
    calls the OCR fallback. If users request OCR and do not explicitly choose a
    backend, prefer the same auto reader used by the build pipeline.
    """
    if requested_backend:
        return requested_backend
    return "pymupdf4llm" if enable_ocr else "pymupdf_fast"

def print_ocr_runtime_note(ocr_status: dict, backend: str) -> None:
    """Print a concise runtime note before the long PDF loop starts."""
    if not ocr_status.get("ocr_requested"):
        return
    if backend == "pymupdf_fast":
        print(
            "[ocr] --enable-ocr was requested, but --backend pymupdf_fast "
            "does not run OCR fallback.",
            file=sys.stderr,
            flush=True,
        )
    if not ocr_status.get("ocr_available"):
        print(
            "[ocr] OCRmyPDF executable not found in PATH; scan-like PDFs will "
            "remain NEED_MANUAL_REVIEW until OCRmyPDF/Tesseract/Ghostscript "
            "are installed.",
            file=sys.stderr,
            flush=True,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_bank_year_summary(
    sample_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    years: Iterable[int],
) -> dict:
    sample = sample_df[sample_df["include_in_main_sample"].map(_truthy)].copy()
    expected = {
        (str(row["bank_id"]).strip().upper(), int(year))
        for _, row in sample.iterrows()
        for year in years
    }
    present = {
        (str(row["bank_id"]).strip().upper(), int(row["year"]))
        for _, row in inventory_df.iterrows()
    }
    missing = expected - present
    expected_count = len(expected)
    present_count = len(expected & present)
    missing_count = len(missing)
    return {
        "expected_bank_years": expected_count,
        "present_bank_years": present_count,
        "missing_bank_years": missing_count,
        "missing_rate": round(missing_count / expected_count, 6) if expected_count else 0.0,
    }


def current_missing_bank_year_rows(
    sample_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    years: Iterable[int],
) -> pd.DataFrame:
    sample = sample_df[sample_df["include_in_main_sample"].map(_truthy)].copy()
    present = {
        (str(row["bank_id"]).strip().upper(), int(row["year"]))
        for _, row in inventory_df.iterrows()
    }
    rows = []
    for _, row in sample.iterrows():
        bank_id = str(row["bank_id"]).strip().upper()
        for year in years:
            if (bank_id, int(year)) in present:
                continue
            rows.append({
                "sample_order": row.get("sample_order", ""),
                "bank_id": bank_id,
                "bank_name": row.get("bank_name", ""),
                "common_name": row.get("common_name", ""),
                "year": int(year),
                "missing_reason": "not_in_current_inventory",
                "notes": row.get("notes", ""),
            })
    return pd.DataFrame(rows)


def check_inventory_files(
    inventory_df: pd.DataFrame,
    preferred_backend: str = "pymupdf4llm",
    enable_ocr: bool = False,
    skip_checksum: bool = False,
    checkpoint_callback: Callable[[pd.DataFrame], None] | None = None,
    checkpoint_every: int = 0,
) -> pd.DataFrame:
    rows = []
    cfg = DocumentReaderConfig(
        preferred_backend=preferred_backend,
        enable_ocr=enable_ocr,
    )
    total = len(inventory_df)

    for position, (_, record) in enumerate(inventory_df.iterrows(), start=1):
        pdf_path = _as_project_path(record["pdf_path"])
        expected_sha = str(record.get("raw_sha256", "") or "").strip().lower()
        exists = pdf_path.exists()
        actual_sha = ""
        checksum_ok = False
        file_size_bytes = 0

        if exists:
            file_size_bytes = pdf_path.stat().st_size
            if skip_checksum:
                checksum_ok = True
            else:
                actual_sha = sha256_file(pdf_path)
                checksum_ok = bool(expected_sha) and actual_sha.lower() == expected_sha

        quality = {
            "pdf_backend": "",
            "extraction_status": "FILE_MISSING" if not exists else "NOT_CHECKED",
            "ocr_used": False,
            "ocr_pdf_path": "",
            "scan_detected": False,
            "text_coverage": 0.0,
            "garbled_ratio": 0.0,
            "raw_total_words": 0,
            "num_pages": 0,
            "text_pages_ge10": 0,
            "text_pages_ge50": 0,
            "text_page_share_ge10": 0.0,
            "words_per_physical_page": 0.0,
            "median_words_per_page": 0.0,
            "extraction_warnings": "",
            "read_error": "",
        }

        if exists:
            try:
                if preferred_backend == "pymupdf_fast":
                    result = read_pdf_fast_pymupdf(pdf_path)
                else:
                    result = read_pdf_auto(str(pdf_path), cfg)
                quality.update({
                    "pdf_backend": result.backend,
                    "extraction_status": result.status,
                    "ocr_used": result.ocr_used,
                    "ocr_pdf_path": getattr(result, "ocr_pdf_path", ""),
                    "scan_detected": result.scan_detected,
                    "text_coverage": result.text_coverage,
                    "garbled_ratio": result.garbled_ratio,
                    "raw_total_words": result.raw_total_words,
                    "num_pages": result.num_pages,
                    "text_pages_ge10": getattr(result, "text_pages_ge10", 0),
                    "text_pages_ge50": getattr(result, "text_pages_ge50", 0),
                    "text_page_share_ge10": getattr(
                        result, "text_page_share_ge10", 0.0
                    ),
                    "words_per_physical_page": getattr(
                        result, "words_per_physical_page", 0.0
                    ),
                    "median_words_per_page": getattr(
                        result, "median_words_per_page", 0.0
                    ),
                    "extraction_warnings": " | ".join(result.extraction_warnings),
                })
            except Exception as exc:
                quality["extraction_status"] = "READ_ERROR"
                quality["read_error"] = str(exc)

        rows.append({
            "bank_id": record["bank_id"],
            "year": int(record["year"]),
            "pdf_path": record["pdf_path"],
            "file_exists": exists,
            "file_size_bytes": file_size_bytes,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "checksum_ok": checksum_ok,
            **quality,
        })

        if checkpoint_callback and checkpoint_every > 0 and (
            position % checkpoint_every == 0 or position == total
        ):
            checkpoint_callback(pd.DataFrame(rows))

        if position % 25 == 0 or position == total:
            print(f"Checked {position}/{total} PDFs", flush=True)

    return pd.DataFrame(rows)

def select_inventory_rows_for_quality_refresh(
    inventory_df: pd.DataFrame,
    existing_quality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return inventory rows whose existing quality status still needs work."""
    refresh_keys = {
        _quality_key(row["bank_id"], row["year"])
        for _, row in existing_quality_df.iterrows()
        if _review_reason(row)
    }
    if not refresh_keys:
        return inventory_df.iloc[0:0].copy()

    selected = inventory_df[
        [
            _quality_key(row["bank_id"], row["year"]) in refresh_keys
            for _, row in inventory_df.iterrows()
        ]
    ].copy()
    return selected

def merge_quality_refresh_rows(
    existing_quality_df: pd.DataFrame,
    refreshed_quality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Replace refreshed bank-year rows while preserving existing row order."""
    if refreshed_quality_df.empty:
        return existing_quality_df.copy()

    refreshed_keys = {
        _quality_key(row["bank_id"], row["year"])
        for _, row in refreshed_quality_df.iterrows()
    }
    existing = existing_quality_df.copy()
    existing["_quality_key"] = [
        _quality_key(row["bank_id"], row["year"]) for _, row in existing.iterrows()
    ]
    existing["_quality_order"] = range(len(existing))

    refreshed = refreshed_quality_df.copy()
    order_by_key = dict(zip(existing["_quality_key"], existing["_quality_order"]))
    next_order = len(existing)
    refreshed["_quality_key"] = [
        _quality_key(row["bank_id"], row["year"]) for _, row in refreshed.iterrows()
    ]
    refreshed["_quality_order"] = [
        order_by_key.get(key, next_order + idx)
        for idx, key in enumerate(refreshed["_quality_key"])
    ]

    kept = existing[~existing["_quality_key"].isin(refreshed_keys)]
    merged = (
        pd.concat([kept, refreshed], ignore_index=True)
        .sort_values("_quality_order")
        .drop(columns=["_quality_key", "_quality_order"])
        .reset_index(drop=True)
    )
    return merged


def _quality_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _review_reason(row: pd.Series) -> str:
    status = str(row.get("extraction_status", "")).strip().upper()
    warnings = str(row.get("extraction_warnings", "") or "").upper()
    if not _quality_bool(row.get("file_exists", True)):
        return "file_missing"
    if status == "FILE_MISSING":
        return "file_missing"
    if status == "READ_ERROR":
        return "read_error"
    if status == "NEED_MANUAL_REVIEW" and "NEEDS_OCR" in warnings:
        return "needs_ocr"
    if status == "NEED_MANUAL_REVIEW" and "POOR_TEXT_LAYER" in warnings:
        return "poor_text_layer"
    if status == "NEED_MANUAL_REVIEW":
        return "manual_review"
    return ""


def _recommended_action(reason: str) -> str:
    actions = {
        "file_missing": "Locate or redownload the raw PDF, then update report_inventory.csv.",
        "read_error": "Open the PDF manually, replace corrupt files through metadata, or exclude with documented reason.",
        "needs_ocr": "Install OCRmyPDF/Tesseract/Ghostscript and rerun quality check with --enable-ocr.",
        "poor_text_layer": "Rerun with --enable-ocr or inspect extracted text before using this bank-year.",
        "manual_review": "Inspect extracted text and decide whether to rerun OCR, exclude, or accept with notes.",
    }
    return actions.get(reason, "Inspect this bank-year before using it in the main panel.")


def extraction_review_rows(quality_df: pd.DataFrame) -> pd.DataFrame:
    """Return bank-year rows that should be reviewed before panel construction."""
    if quality_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in quality_df.iterrows():
        reason = _review_reason(row)
        if not reason:
            continue
        output = row.to_dict()
        output["review_reason"] = reason
        output["recommended_action"] = _recommended_action(reason)
        rows.append(output)
    return pd.DataFrame(rows)


def read_pdf_fast_pymupdf(pdf_path: Path):
    """Fast text-layer QA with PyMuPDF without layout parsing or OCR."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("pip install pymupdf") from exc

    with fitz.open(str(pdf_path)) as doc:
        pages = [
            PageText(idx + 1, page.get_text("text") or "", "pymupdf_fast")
            for idx, page in enumerate(doc)
        ]
        page_count = doc.page_count
    quality = assess_text_quality(pages, num_pages=page_count)

    class FastReadResult:
        backend = "pymupdf_fast"
        status = "OK" if quality.status == "OK" else "NEED_MANUAL_REVIEW"
        ocr_used = False
        scan_detected = quality.scan_detected
        text_coverage = quality.text_coverage
        garbled_ratio = quality.garbled_ratio
        raw_total_words = quality.total_words
        num_pages = page_count
        extraction_warnings = [] if quality.status == "OK" else [quality.status]

    return FastReadResult()


def build_summary(
    quality_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    years: Iterable[int],
    ocr_status: dict | None = None,
) -> dict:
    status_counts = quality_df["extraction_status"].value_counts(dropna=False).to_dict()
    backend_counts = quality_df["pdf_backend"].value_counts(dropna=False).to_dict()
    missing_reason_counts = (
        missing_df["missing_reason"].value_counts(dropna=False).to_dict()
        if not missing_df.empty and "missing_reason" in missing_df.columns
        else {}
    )
    expected_summary = expected_bank_year_summary(sample_df, inventory_df, years)
    return {
        "run_date": date.today().isoformat(),
        **expected_summary,
        "inventory_rows": int(len(inventory_df)),
        "checked_pdf_rows": int(len(quality_df)),
        "files_missing_on_disk": int((~quality_df["file_exists"]).sum()),
        "checksum_failures": int((~quality_df["checksum_ok"]).sum()),
        "read_errors": int((quality_df["extraction_status"] == "READ_ERROR").sum()),
        "manual_review_rows": int((quality_df["extraction_status"] == "NEED_MANUAL_REVIEW").sum()),
        "scan_detected_rows": int(quality_df["scan_detected"].sum()),
        "total_pdf_size_mb": round(float(quality_df["file_size_bytes"].sum()) / 1024 / 1024, 2),
        "status_counts": status_counts,
        "backend_counts": backend_counts,
        "missing_reason_counts": missing_reason_counts,
        **(ocr_status or ocr_dependency_status(enable_ocr=False)),
    }

def write_quality_outputs(
    quality_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    years: Iterable[int],
    output_path: Path,
    summary_path: Path,
    review_output_path: Path,
    current_missing_path: Path,
    ocr_status: dict,
) -> dict:
    """Write quality artifacts atomically enough for resumable long OCR runs."""
    summary = build_summary(
        quality_df,
        sample_df,
        inventory_df,
        missing_df,
        years,
        ocr_status=ocr_status,
    )
    current_missing_df = current_missing_bank_year_rows(sample_df, inventory_df, years)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    review_output_path.parent.mkdir(parents=True, exist_ok=True)
    current_missing_path.parent.mkdir(parents=True, exist_ok=True)

    quality_tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    review_tmp = review_output_path.with_suffix(review_output_path.suffix + ".tmp")
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    current_missing_tmp = current_missing_path.with_suffix(current_missing_path.suffix + ".tmp")

    quality_df.to_csv(quality_tmp, index=False, encoding="utf-8-sig")
    extraction_review_rows(quality_df).to_csv(review_tmp, index=False, encoding="utf-8-sig")
    summary_tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    current_missing_df.to_csv(current_missing_tmp, index=False, encoding="utf-8-sig")

    quality_tmp.replace(output_path)
    review_tmp.replace(review_output_path)
    summary_tmp.replace(summary_path)
    current_missing_tmp.replace(current_missing_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check annual report PDF quality.")
    parser.add_argument("--inventory", default=str(PROJECT_ROOT / "data" / "metadata" / "report_inventory.csv"))
    parser.add_argument("--sample", default=str(PROJECT_ROOT / "data" / "metadata" / "bank_sample_2015_2025.csv"))
    parser.add_argument(
        "--missing",
        default=str(PROJECT_ROOT / "data" / "metadata" / "report_inventory_missing_2015_2025.csv"),
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "logs" / "downloaded_pdf_quality.csv"))
    parser.add_argument("--summary", default=str(PROJECT_ROOT / "results" / "logs" / "downloaded_pdf_quality_summary.json"))
    parser.add_argument(
        "--review-output",
        default=str(PROJECT_ROOT / "data" / "metadata" / "report_inventory_needs_review.csv"),
    )
    parser.add_argument(
        "--current-missing-output",
        default=str(PROJECT_ROOT / "data" / "metadata" / "report_inventory_missing_current_2015_2025.csv"),
    )
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--backend",
        default=None,
        choices=["pymupdf_fast", "pymupdf4llm", "pdfplumber", "pypdf"],
        help=(
            "PDF backend for quality checks. Defaults to pymupdf_fast, or "
            "pymupdf4llm when --enable-ocr is set so OCR fallback can run."
        ),
    )
    parser.add_argument("--enable-ocr", action="store_true", help="Allow OCR fallback for scan-like PDFs.")
    parser.add_argument(
        "--only-review",
        action="store_true",
        help=(
            "Reuse the existing --output quality log and recheck only rows "
            "that currently need review. Useful after a fast full pass."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Write partial quality outputs every N checked PDFs. Defaults to "
            "1 for --only-review and disabled for full runs."
        ),
    )
    parser.add_argument("--skip-checksum", action="store_true", help="Skip SHA256 verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory_df = pd.read_csv(_as_project_path(args.inventory))
    sample_df = pd.read_csv(_as_project_path(args.sample))
    missing_path = _as_project_path(args.missing)
    missing_df = pd.read_csv(missing_path) if missing_path.exists() else pd.DataFrame()
    years = range(args.start_year, args.end_year + 1)
    backend = resolve_quality_backend(args.backend, args.enable_ocr)
    ocr_status = ocr_dependency_status(args.enable_ocr)
    print_ocr_runtime_note(ocr_status, backend)
    output_path = _as_project_path(args.output)
    summary_path = _as_project_path(args.summary)
    review_output_path = _as_project_path(args.review_output)
    current_missing_path = _as_project_path(args.current_missing_output)

    existing_quality_df = pd.DataFrame()
    quality_inventory_df = inventory_df
    if args.only_review:
        if not output_path.exists():
            raise FileNotFoundError(
                f"--only-review needs an existing quality log at {output_path}. "
                "Run once without --only-review first."
            )
        existing_quality_df = pd.read_csv(output_path)
        quality_inventory_df = select_inventory_rows_for_quality_refresh(
            inventory_df,
            existing_quality_df,
        )
        print(
            f"[only-review] Rechecking {len(quality_inventory_df)}/"
            f"{len(inventory_df)} inventory rows from existing quality log.",
            flush=True,
        )

    checkpoint_every = args.checkpoint_every
    if args.only_review and checkpoint_every <= 0:
        checkpoint_every = 1

    def checkpoint_quality(partial_refreshed_quality_df: pd.DataFrame) -> None:
        if args.only_review:
            checkpoint_quality_df = merge_quality_refresh_rows(
                existing_quality_df,
                partial_refreshed_quality_df,
            )
        else:
            checkpoint_quality_df = partial_refreshed_quality_df
        checkpoint_summary = write_quality_outputs(
            checkpoint_quality_df,
            sample_df,
            inventory_df,
            missing_df,
            years,
            output_path,
            summary_path,
            review_output_path,
            current_missing_path,
            ocr_status,
        )
        print(
            "[checkpoint] "
            f"checked={len(partial_refreshed_quality_df)}/{len(quality_inventory_df)} "
            f"manual_review_rows={checkpoint_summary['manual_review_rows']}",
            flush=True,
        )

    refreshed_quality_df = check_inventory_files(
        quality_inventory_df,
        preferred_backend=backend,
        enable_ocr=args.enable_ocr,
        skip_checksum=args.skip_checksum,
        checkpoint_callback=checkpoint_quality if checkpoint_every > 0 else None,
        checkpoint_every=checkpoint_every,
    )
    quality_df = (
        merge_quality_refresh_rows(existing_quality_df, refreshed_quality_df)
        if args.only_review
        else refreshed_quality_df
    )
    summary = write_quality_outputs(
        quality_df,
        sample_df,
        inventory_df,
        missing_df,
        years,
        output_path,
        summary_path,
        review_output_path,
        current_missing_path,
        ocr_status,
    )

    print(f"Quality rows -> {output_path}")
    print(f"Summary -> {summary_path}")
    print(f"Needs-review rows -> {review_output_path}")
    print(f"Current missing bank-years -> {current_missing_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
