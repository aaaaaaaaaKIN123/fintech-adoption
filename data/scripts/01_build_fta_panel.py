"""
Build the FinTech Adoption panel from annual-report PDFs.

Input inventory columns:
    bank_id, year, pdf_path

Default output:
    data/analysis_output/FTA_panel.csv
    data/analysis_output/FTA_records.json
    data/analysis_output/<bank>_<year>_segments.jsonl
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fintech_adoption.config import PipelineConfig  # noqa: E402
from fintech_adoption.llm_adjudicator import LLMPreflightError, run_llm_preflight  # noqa: E402
from fintech_adoption.pipeline import run_batch  # noqa: E402

BLOCKING_EXTRACTION_STATUSES = {"FILE_MISSING", "READ_ERROR", "NEED_MANUAL_REVIEW"}


class ExtractionQualityGateError(RuntimeError):
    """Raised when selected reports have extraction issues that need review."""


def _resolve_path(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def _quality_key(bank_id, year) -> Tuple[str, int]:
    return str(bank_id).strip().upper(), int(year)


def _quality_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _format_gate_issue(row: dict) -> str:
    bank_id = str(row.get("bank_id", "")).strip()
    year = int(row.get("year", 0))
    status = str(row.get("extraction_status", "")).strip() or "MISSING_QUALITY_LOG_ROW"
    warnings = str(row.get("extraction_warnings", "") or "").strip()
    if warnings:
        return f"{bank_id} {year} ({status}; {warnings})"
    return f"{bank_id} {year} ({status})"


def enforce_extraction_quality_gate(
    file_map: Dict[Tuple[str, int], str],
    quality_log_path: str,
    allow_manual_review: bool = False,
    review_output_path: str | None = None,
) -> None:
    """Fail before panel construction if selected PDFs still need extraction review."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ExtractionQualityGateError("pandas is required for the extraction quality gate.") from exc

    path = Path(_resolve_path(quality_log_path))
    if not path.exists():
        raise ExtractionQualityGateError(
            "Extraction quality log not found: "
            f"{path}. Run data/scripts/04_check_report_pdf_quality.py first "
            "or pass --skip-extraction-gate for exploratory runs."
        )

    quality_df = pd.read_csv(path)
    selected = {_quality_key(bank_id, year) for bank_id, year in file_map}
    quality_df["_gate_key"] = [
        _quality_key(row["bank_id"], row["year"]) for _, row in quality_df.iterrows()
    ]
    selected_quality = quality_df[quality_df["_gate_key"].isin(selected)].copy()
    seen = set(selected_quality["_gate_key"])

    blocking_rows = []
    for bank_id, year in sorted(selected - seen):
        blocking_rows.append({
            "bank_id": bank_id,
            "year": year,
            "extraction_status": "MISSING_QUALITY_LOG_ROW",
            "extraction_warnings": "Selected report has no row in the extraction quality log.",
        })

    for _, row in selected_quality.iterrows():
        status = str(row.get("extraction_status", "")).strip().upper()
        if allow_manual_review and status == "NEED_MANUAL_REVIEW":
            continue
        if status in BLOCKING_EXTRACTION_STATUSES:
            blocking_rows.append(row.drop(labels=["_gate_key"], errors="ignore").to_dict())
            continue
        if "file_exists" in row and not _quality_bool(row.get("file_exists", True)):
            blocking_rows.append(row.drop(labels=["_gate_key"], errors="ignore").to_dict())

    if not blocking_rows:
        return

    if review_output_path:
        review_path = Path(_resolve_path(review_output_path))
        review_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(blocking_rows).to_csv(review_path, index=False, encoding="utf-8-sig")

    examples = "\n".join(f"- {_format_gate_issue(row)}" for row in blocking_rows[:10])
    extra = "" if len(blocking_rows) <= 10 else f"\n... and {len(blocking_rows) - 10} more"
    raise ExtractionQualityGateError(
        "Extraction quality gate failed. Resolve these bank-years before building "
        "the main FTA panel, or pass --allow-manual-review for exploratory runs:\n"
        f"{examples}{extra}"
    )


def load_report_inventory(path: str) -> Dict[Tuple[str, int], str]:
    inventory_path = Path(path)
    if not inventory_path.is_absolute():
        inventory_path = ROOT / inventory_path
    if not inventory_path.exists():
        raise FileNotFoundError(f"Report inventory not found: {inventory_path}")

    file_map: Dict[Tuple[str, int], str] = {}
    with open(inventory_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"bank_id", "year", "pdf_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Inventory missing required columns: {sorted(missing)}")

        for row in reader:
            bank_id = str(row["bank_id"]).strip()
            year = int(row["year"])
            pdf_path = _resolve_path(str(row["pdf_path"]).strip())
            if bank_id and pdf_path:
                file_map[(bank_id, year)] = pdf_path
    return file_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FTA_it panel from report inventory.")
    parser.add_argument(
        "--inventory",
        default=str(ROOT / "data" / "metadata" / "report_inventory.csv"),
        help="CSV with bank_id, year, pdf_path columns.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to PipelineConfig data/analysis_output.",
    )
    parser.add_argument("--mock-llm", action="store_true", help="Use deterministic mock LLM.")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM adjudication.")
    parser.add_argument("--api-key", default=None, help="Anthropic/OpenAI API key.")
    parser.add_argument("--llm-provider", default=None, help="Override LLM provider.")
    parser.add_argument("--llm-model", default=None, help="Override LLM model.")
    parser.add_argument("--top-k", type=int, default=None, help="Override Tier-1 top-K.")
    parser.add_argument("--alpha", type=float, default=None, help="Override Tier-1 alpha.")
    parser.add_argument(
        "--pdf-backend-timeout-seconds",
        type=int,
        default=180,
        help="Seconds before timing out PyMuPDF4LLM and falling back to PyMuPDF fast text.",
    )
    parser.add_argument(
        "--quality-log",
        default=str(ROOT / "results" / "logs" / "downloaded_pdf_quality.csv"),
        help="Pre-build PDF quality CSV from data/scripts/04_check_report_pdf_quality.py.",
    )
    parser.add_argument(
        "--review-output",
        default=str(ROOT / "data" / "metadata" / "report_inventory_needs_review.csv"),
        help="Where to write selected rows blocked by the extraction quality gate.",
    )
    parser.add_argument(
        "--skip-extraction-gate",
        action="store_true",
        help="Skip pre-build PDF quality gating for exploratory runs.",
    )
    parser.add_argument(
        "--allow-manual-review",
        action="store_true",
        help="Allow NEED_MANUAL_REVIEW rows while still blocking missing/read-error rows.",
    )
    parser.add_argument(
        "--allow-partial-reports",
        action="store_true",
        help="Allow exploratory output when some selected reports fail preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig()
    if args.output:
        cfg.output_dir = _resolve_path(args.output)
    if args.top_k is not None:
        cfg.tier1.top_k = args.top_k
    if args.alpha is not None:
        cfg.tier1.alpha = args.alpha
    if args.llm_provider:
        cfg.llm.provider = args.llm_provider
    if args.llm_model:
        cfg.llm.model = args.llm_model
    cfg.extraction.backend_timeout_seconds = args.pdf_backend_timeout_seconds
    cfg.run_llm_adjudication = not args.no_llm
    cfg.allow_partial_reports = args.allow_partial_reports

    file_map = load_report_inventory(args.inventory)
    if not args.skip_extraction_gate:
        try:
            enforce_extraction_quality_gate(
                file_map,
                args.quality_log,
                allow_manual_review=args.allow_manual_review,
                review_output_path=args.review_output,
            )
        except ExtractionQualityGateError as exc:
            print(f"[extraction-gate] {exc}", file=sys.stderr)
            sys.exit(3)

    if cfg.llm.provider == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    else:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if cfg.run_llm_adjudication and not args.mock_llm:
        print(f"[preflight] Checking {cfg.llm.provider} model {cfg.llm.model} ...", flush=True)
        try:
            run_llm_preflight(cfg.llm, api_key=api_key)
        except LLMPreflightError as exc:
            print(f"[preflight] {exc}", file=sys.stderr)
            sys.exit(2)
        print("[preflight] LLM API ready.")
    run_batch(file_map, cfg, api_key=api_key, mock_llm=args.mock_llm)


if __name__ == "__main__":
    main()
