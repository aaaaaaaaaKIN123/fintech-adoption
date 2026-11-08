"""
document_reader.py - Robust PDF extraction layer for annual reports.

The reader is backend-pluggable:
1. PyMuPDF4LLM when available.
2. OCRmyPDF + PyMuPDF4LLM when the initial extraction looks scanned/sparse.
3. pdfplumber fallback.
4. pypdf fallback.

Raw PDFs are never modified. OCR output is written to data/processed/ocr.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    page_num: int
    text: str
    backend: str
    markdown: str = ""
    word_count: int = 0

    def __post_init__(self):
        if not self.markdown:
            self.markdown = self.text
        self.word_count = len(self.text.split())


@dataclass
class TextQuality:
    text_coverage: float
    scan_detected: bool
    garbled_ratio: float
    total_chars: int
    total_words: int
    status: str
    page_count: int = 0
    text_pages_ge10: int = 0
    text_pages_ge50: int = 0
    text_page_share_ge10: float = 0.0
    words_per_physical_page: float = 0.0
    median_words_per_page: float = 0.0
    page_quality: List[dict] = field(default_factory=list)


@dataclass
class DocumentReaderConfig:
    preferred_backend: str = "pymupdf4llm"
    fallback_backend: str = "pymupdf_fast"
    backend_timeout_seconds: int = 180
    enable_ocr: bool = True
    ocr_engine: str = "ocrmypdf"
    ocr_languages: str = "vie+eng"
    ocr_output_dir: str = "data/processed/ocr"
    min_chars_per_page: int = 200
    min_words_total: int = 50
    max_garbled_ratio: float = 0.08
    min_words_per_text_page: int = 10
    min_text_page_share: float = 0.50
    long_document_min_pages: int = 30
    min_words_per_page_long_document: float = 80.0
    ocr_mode: str = "skip"
    ocr_invalidate_digital_signatures: bool = False
    ocr_jobs: int = 2
    ocr_timeout_seconds: int = 1800


@dataclass
class DocumentReadResult:
    pdf_path: str
    backend: str
    pages: List[PageText] = field(default_factory=list)
    status: str = "OK"
    ocr_used: bool = False
    ocr_engine: str = ""
    ocr_pdf_path: str = ""
    scan_detected: bool = False
    text_coverage: float = 0.0
    garbled_ratio: float = 0.0
    raw_total_words: int = 0
    num_pages: int = 0
    extraction_warnings: List[str] = field(default_factory=list)
    text_pages_ge10: int = 0
    text_pages_ge50: int = 0
    text_page_share_ge10: float = 0.0
    words_per_physical_page: float = 0.0
    median_words_per_page: float = 0.0
    page_quality: List[dict] = field(default_factory=list)


def assess_text_quality(
    pages: List[PageText],
    num_pages: Optional[int] = None,
    cfg: Optional[DocumentReaderConfig] = None,
) -> TextQuality:
    """Assess whether extracted text is usable or likely needs OCR."""
    cfg = cfg or DocumentReaderConfig()
    page_count = max(num_pages or len(pages), 1)
    total_chars = sum(len(page.text.strip()) for page in pages)
    total_words = sum(page.word_count for page in pages)

    page_map: dict[int, PageText] = {}
    for page in pages:
        if not 1 <= int(page.page_num) <= page_count:
            continue
        existing = page_map.get(int(page.page_num))
        if existing is None or page.word_count > existing.word_count:
            page_map[int(page.page_num)] = page

    page_word_counts = [
        page_map[page_num].word_count if page_num in page_map else 0
        for page_num in range(1, page_count + 1)
    ]
    page_char_counts = [
        len(page_map[page_num].text.strip()) if page_num in page_map else 0
        for page_num in range(1, page_count + 1)
    ]
    text_pages_ge10 = sum(
        count >= cfg.min_words_per_text_page for count in page_word_counts
    )
    text_pages_ge50 = sum(count >= 50 for count in page_word_counts)
    text_page_share_ge10 = text_pages_ge10 / page_count
    words_per_physical_page = total_words / page_count
    median_words_per_page = float(statistics.median(page_word_counts))
    text_coverage = text_page_share_ge10

    bad_chars = sum(1 for page in pages for ch in page.text if ch == "\ufffd" or ord(ch) < 9)
    garbled_ratio = bad_chars / max(total_chars, 1)
    sparse_page_coverage = (
        page_count >= 10 and text_page_share_ge10 < cfg.min_text_page_share
    )
    sparse_long_document = (
        page_count >= cfg.long_document_min_pages
        and words_per_physical_page < cfg.min_words_per_page_long_document
    )
    scan_detected = (
        total_words < cfg.min_words_total
        or sparse_page_coverage
        or sparse_long_document
    )

    if scan_detected:
        status = "NEEDS_OCR"
    elif garbled_ratio > cfg.max_garbled_ratio:
        status = "POOR_TEXT_LAYER"
    else:
        status = "OK"

    page_quality = []
    for page_num, (word_count, char_count) in enumerate(
        zip(page_word_counts, page_char_counts),
        start=1,
    ):
        if word_count >= cfg.min_words_per_text_page:
            page_status = "TEXT_OK"
        elif word_count > 0:
            page_status = "SPARSE_TEXT"
        else:
            page_status = "NO_TEXT"
        page_quality.append(
            {
                "page_num": page_num,
                "word_count": int(word_count),
                "char_count": int(char_count),
                "status": page_status,
            }
        )

    return TextQuality(
        text_coverage=round(text_coverage, 4),
        scan_detected=scan_detected,
        garbled_ratio=round(garbled_ratio, 4),
        total_chars=total_chars,
        total_words=total_words,
        status=status,
        page_count=page_count,
        text_pages_ge10=int(text_pages_ge10),
        text_pages_ge50=int(text_pages_ge50),
        text_page_share_ge10=round(text_page_share_ge10, 4),
        words_per_physical_page=round(words_per_physical_page, 2),
        median_words_per_page=round(median_words_per_page, 2),
        page_quality=page_quality,
    )


def extract_with_pymupdf4llm(pdf_path: str) -> List[PageText]:
    """Extract text/markdown with PyMuPDF4LLM if installed."""
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise ImportError("pip install pymupdf4llm") from exc

    try:
        pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    except TypeError:
        markdown = pymupdf4llm.to_markdown(pdf_path)
        return [PageText(1, markdown or "", "pymupdf4llm", markdown=markdown or "")]

    results: List[PageText] = []
    for idx, page in enumerate(pages):
        if isinstance(page, dict):
            text = page.get("text") or page.get("markdown") or ""
            page_num = int(page.get("metadata", {}).get("page", idx + 1))
        else:
            text = str(page)
            page_num = idx + 1
        if text.strip():
            results.append(PageText(page_num, text, "pymupdf4llm", markdown=text))
    return results


def extract_with_pymupdf_fast(pdf_path: str) -> List[PageText]:
    """Extract plain text page-by-page with PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("pip install pymupdf") from exc

    pages: List[PageText] = []
    with fitz.open(str(pdf_path)) as doc:
        for idx, page in enumerate(doc):
            text = page.get_text("text") or ""
            if text.strip():
                pages.append(PageText(idx + 1, text, "pymupdf_fast"))
    return pages


def extract_with_pdfplumber_pages(pdf_path: str) -> List[PageText]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pip install pdfplumber") from exc

    pages: List[PageText] = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            if text.strip():
                pages.append(PageText(idx + 1, text, "pdfplumber"))
    return pages


def extract_with_pypdf_pages(pdf_path: str) -> List[PageText]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pip install pypdf") from exc

    reader = PdfReader(pdf_path)
    pages: List[PageText] = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(PageText(idx + 1, text, "pypdf"))
    return pages


def get_pdf_page_count(pdf_path: str) -> Optional[int]:
    """Return the physical PDF page count when a lightweight reader is available."""
    try:
        import fitz

        with fitz.open(str(pdf_path)) as doc:
            return int(doc.page_count)
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        return len(PdfReader(pdf_path).pages)
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_cache_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value).strip("-") or "default"


def _ocr_cache_paths(
    input_path: Path,
    output_dir: str,
    languages: str,
    mode: str,
) -> tuple[Path, Path, str]:
    input_sha256 = _sha256_file(input_path)
    language_token = _safe_cache_token(languages)
    mode_token = _safe_cache_token(mode)
    output_path = (
        Path(output_dir)
        / f"{input_path.stem}_{input_sha256}_{language_token}_{mode_token}_ocr.pdf"
    )
    manifest_path = output_path.with_suffix(".ocr.json")
    return output_path, manifest_path, input_sha256


def _valid_ocr_cache(
    output_path: Path,
    manifest_path: Path,
    input_sha256: str,
    languages: str,
    mode: str,
) -> bool:
    if not output_path.exists() or output_path.stat().st_size == 0 or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        manifest.get("input_sha256") == input_sha256
        and manifest.get("languages") == languages
        and manifest.get("ocr_engine") == "ocrmypdf"
        and manifest.get("mode") == mode
    )


def run_ocrmypdf(
    input_pdf: str,
    output_dir: str,
    languages: str = "vie+eng",
    mode: str = "skip",
    invalidate_digital_signatures: bool = False,
    jobs: int = 2,
    timeout_seconds: int | None = None,
) -> str:
    """Run OCRmyPDF CLI and return the OCR PDF path."""
    if mode not in {"skip", "force", "redo"}:
        raise ValueError(f"Unsupported OCR mode: {mode}")
    input_path = Path(input_pdf)
    output_path, manifest_path, input_sha256 = _ocr_cache_paths(
        input_path,
        output_dir,
        languages,
        mode,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _valid_ocr_cache(
        output_path,
        manifest_path,
        input_sha256,
        languages,
        mode,
    ):
        return str(output_path)

    if output_path.exists():
        output_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    cmd = [
        "ocrmypdf",
        "-l",
        languages,
        "--output-type",
        "pdf",
        "--jobs",
        str(max(1, int(jobs))),
        "--optimize",
        "0",
        "--deskew",
    ]
    cmd.append(
        {
            "skip": "--skip-text",
            "force": "--force-ocr",
            "redo": "--redo-ocr",
        }[mode]
    )
    if invalidate_digital_signatures:
        cmd.append("--invalidate-digital-signatures")
    cmd.extend([str(input_path), str(output_path)])
    run_kwargs = {"capture_output": True, "text": True, "check": False}
    if timeout_seconds is not None:
        run_kwargs["timeout"] = int(timeout_seconds)
    completed = subprocess.run(cmd, **run_kwargs)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"OCRmyPDF failed: {message}")
    manifest = {
        "source_pdf": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "input_size_bytes": input_path.stat().st_size,
        "languages": languages,
        "mode": mode,
        "ocr_engine": "ocrmypdf",
        "invalidate_digital_signatures": bool(invalidate_digital_signatures),
        "jobs": int(jobs),
        "optimize": 0,
        "output_pdf": str(output_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def _try_backend_direct(name: str, pdf_path: str) -> List[PageText]:
    if name == "pymupdf4llm":
        return extract_with_pymupdf4llm(pdf_path)
    if name == "pymupdf_fast":
        return extract_with_pymupdf_fast(pdf_path)
    if name == "pdfplumber":
        return extract_with_pdfplumber_pages(pdf_path)
    if name == "pypdf":
        return extract_with_pypdf_pages(pdf_path)
    raise ValueError(f"Unknown PDF backend: {name}")


def _backend_worker(name: str, pdf_path: str, queue) -> None:
    try:
        queue.put(("ok", _try_backend_direct(name, pdf_path)))
    except Exception as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_backend_with_timeout(name: str, pdf_path: str, timeout_seconds: int) -> List[PageText]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_backend_worker, args=(name, pdf_path, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise TimeoutError(f"{name} timed out after {timeout_seconds}s")
    if queue.empty():
        raise RuntimeError(f"{name} exited without returning text")
    status, payload = queue.get()
    if status == "ok":
        return payload
    raise RuntimeError(str(payload))


def _try_backend(
    name: str,
    pdf_path: str,
    timeout_seconds: int | None = None,
) -> List[PageText]:
    if name == "pymupdf4llm" and timeout_seconds and timeout_seconds > 0:
        return _run_backend_with_timeout(name, pdf_path, int(timeout_seconds))
    return _try_backend_direct(name, pdf_path)


def merge_page_texts(
    sources: List[tuple[str, List[PageText]]],
) -> tuple[List[PageText], str]:
    """Keep the richest available extraction for each physical page.

    Backends often complement one another: layout-aware extraction can recover
    image text while the fast native layer can retain born-digital paragraphs.
    Selecting page-by-page avoids concatenating duplicate text.
    """
    selected: dict[int, PageText] = {}
    source_names: List[str] = []
    for source_name, pages in sources:
        if source_name not in source_names:
            source_names.append(source_name)
        for page in pages:
            page_num = int(page.page_num)
            existing = selected.get(page_num)
            if existing is None or page.word_count > existing.word_count:
                selected[page_num] = page
    merged = [selected[page_num] for page_num in sorted(selected)]
    if not source_names:
        backend = "none"
    elif len(source_names) == 1:
        backend = source_names[0]
    else:
        backend = "pagewise_best[" + ",".join(source_names) + "]"
    return merged, backend


def read_pdf_auto(
    pdf_path: str,
    cfg: Optional[DocumentReaderConfig] = None,
) -> DocumentReadResult:
    """Read a PDF with quality-aware backend fallback and optional OCR."""
    cfg = cfg or DocumentReaderConfig()
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    warnings: List[str] = []
    physical_page_count = get_pdf_page_count(str(path))
    primary_backends = list(
        dict.fromkeys([cfg.preferred_backend, cfg.fallback_backend])
    )
    remaining_backends = [
        backend
        for backend in ["pdfplumber", "pypdf"]
        if backend not in primary_backends
    ]

    source_results: List[tuple[str, List[PageText]]] = []
    for backend in primary_backends:
        try:
            pages = _try_backend(
                backend,
                str(path),
                timeout_seconds=cfg.backend_timeout_seconds,
            )
            if pages:
                source_results.append((backend, pages))
        except Exception as exc:
            warnings.append(f"{backend}: {exc}")

    best_pages, best_backend = merge_page_texts(source_results)
    best_quality = assess_text_quality(
        best_pages,
        num_pages=physical_page_count,
        cfg=cfg,
    )
    if best_quality.status == "OK":
        return _build_result(
            pdf_path=str(path),
            backend=best_backend,
            pages=best_pages,
            quality=best_quality,
            warnings=warnings,
            num_pages=physical_page_count,
        )

    if cfg.enable_ocr and best_quality.status in {"NEEDS_OCR", "POOR_TEXT_LAYER"}:
        try:
            if cfg.ocr_mode == "adaptive":
                ocr_mode = (
                    "force"
                    if best_quality.text_page_share_ge10 >= cfg.min_text_page_share
                    else "skip"
                )
            else:
                ocr_mode = cfg.ocr_mode
            ocr_pdf = run_ocrmypdf(
                str(path),
                cfg.ocr_output_dir,
                cfg.ocr_languages,
                mode=ocr_mode,
                invalidate_digital_signatures=cfg.ocr_invalidate_digital_signatures,
                jobs=cfg.ocr_jobs,
                timeout_seconds=cfg.ocr_timeout_seconds,
            )
            ocr_pages = extract_with_pymupdf_fast(ocr_pdf)
            ocr_page_count = get_pdf_page_count(ocr_pdf) or physical_page_count
            combined_sources = source_results + [("ocrmypdf+pymupdf_fast", ocr_pages)]
            merged_pages, merged_backend = merge_page_texts(combined_sources)
            merged_quality = assess_text_quality(
                merged_pages,
                num_pages=ocr_page_count,
                cfg=cfg,
            )
            return _build_result(
                pdf_path=str(path),
                backend=merged_backend,
                pages=merged_pages,
                quality=merged_quality,
                warnings=warnings,
                ocr_used=True,
                ocr_engine=cfg.ocr_engine,
                ocr_pdf_path=ocr_pdf,
                force_status=(
                    "OK" if merged_quality.status == "OK" else "NEED_MANUAL_REVIEW"
                ),
                num_pages=ocr_page_count,
            )
        except Exception as exc:
            warnings.append(f"{cfg.ocr_engine}: {exc}")

    for backend in remaining_backends:
        try:
            pages = _try_backend(backend, str(path))
            if pages:
                source_results.append((backend, pages))
        except Exception as exc:
            warnings.append(f"{backend}: {exc}")

    best_pages, best_backend = merge_page_texts(source_results)
    best_quality = assess_text_quality(
        best_pages,
        num_pages=physical_page_count,
        cfg=cfg,
    )
    if best_pages:
        return _build_result(
            pdf_path=str(path),
            backend=best_backend,
            pages=best_pages,
            quality=best_quality,
            warnings=warnings,
            force_status=(
                "OK" if best_quality.status == "OK" else "NEED_MANUAL_REVIEW"
            ),
            num_pages=physical_page_count,
        )

    return DocumentReadResult(
        pdf_path=str(path),
        backend="none",
        pages=[],
        status="NEED_MANUAL_REVIEW",
        scan_detected=True,
        num_pages=physical_page_count or 0,
        extraction_warnings=warnings or ["No backend returned usable text."],
    )


def _build_result(
    pdf_path: str,
    backend: str,
    pages: List[PageText],
    quality: TextQuality,
    warnings: List[str],
    ocr_used: bool = False,
    ocr_engine: str = "",
    ocr_pdf_path: str = "",
    force_status: str = "",
    num_pages: Optional[int] = None,
) -> DocumentReadResult:
    return DocumentReadResult(
        pdf_path=pdf_path,
        backend=backend,
        pages=pages,
        status=force_status or quality.status,
        ocr_used=ocr_used,
        ocr_engine=ocr_engine,
        ocr_pdf_path=ocr_pdf_path,
        scan_detected=quality.scan_detected,
        text_coverage=quality.text_coverage,
        garbled_ratio=quality.garbled_ratio,
        raw_total_words=quality.total_words,
        num_pages=int(num_pages or len(pages)),
        extraction_warnings=list(warnings),
        text_pages_ge10=quality.text_pages_ge10,
        text_pages_ge50=quality.text_pages_ge50,
        text_page_share_ge10=quality.text_page_share_ge10,
        words_per_physical_page=quality.words_per_physical_page,
        median_words_per_page=quality.median_words_per_page,
        page_quality=list(quality.page_quality),
    )
