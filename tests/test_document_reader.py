import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "src" / "fintech_adoption" / "document_reader.py"
SPEC = importlib.util.spec_from_file_location("document_reader", SCRIPT_PATH)
document_reader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = document_reader
SPEC.loader.exec_module(document_reader)


def test_run_ocrmypdf_reuses_checksum_cache_for_same_pdf(tmp_path, monkeypatch):
    input_pdf = tmp_path / "raw" / "bank_report.pdf"
    input_pdf.parent.mkdir()
    input_pdf.write_bytes(b"%PDF-1.4 fake scan bytes")
    output_dir = tmp_path / "ocr"
    calls = []

    def fake_run(cmd, capture_output, text, check):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"%PDF-1.4 fake ocr bytes")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(document_reader.subprocess, "run", fake_run)

    first = document_reader.run_ocrmypdf(str(input_pdf), str(output_dir), "vie+eng")
    second = document_reader.run_ocrmypdf(str(input_pdf), str(output_dir), "vie+eng")

    expected_sha = hashlib.sha256(input_pdf.read_bytes()).hexdigest()
    manifest = json.loads((Path(first).with_suffix(".ocr.json")).read_text(encoding="utf-8"))
    assert first == second
    assert len(calls) == 1
    assert expected_sha in Path(first).name
    assert manifest["input_sha256"] == expected_sha
    assert manifest["languages"] == "vie+eng"


def test_read_pdf_auto_falls_back_to_fast_backend_after_preferred_timeout(tmp_path, monkeypatch):
    pdf_path = tmp_path / "bank_report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake text-layer pdf")

    def fake_try_backend(name, _pdf_path, timeout_seconds=None):
        if name == "pymupdf4llm":
            raise TimeoutError("pymupdf4llm timed out after 1s")
        if name == "pymupdf_fast":
            return [
                document_reader.PageText(
                    page_num=1,
                    text="digital banking adoption " * 30,
                    backend="pymupdf_fast",
                )
            ]
        raise AssertionError(f"unexpected backend: {name}")

    monkeypatch.setattr(document_reader, "_try_backend", fake_try_backend)
    monkeypatch.setattr(document_reader, "get_pdf_page_count", lambda _path: 1)

    result = document_reader.read_pdf_auto(
        str(pdf_path),
        document_reader.DocumentReaderConfig(
            preferred_backend="pymupdf4llm",
            fallback_backend="pymupdf_fast",
            backend_timeout_seconds=1,
            enable_ocr=False,
        ),
    )

    assert result.status == "OK"
    assert result.backend == "pymupdf_fast"
    assert result.pages[0].text.startswith("digital banking adoption")
    assert any("pymupdf4llm timed out" in warning for warning in result.extraction_warnings)


def test_page_level_quality_flags_sparse_long_document():
    pages = [
        document_reader.PageText(
            page_num=page_num,
            text=("digital banking adoption " * 40 if page_num <= 5 else ""),
            backend="test",
        )
        for page_num in range(1, 41)
    ]
    quality = document_reader.assess_text_quality(
        pages,
        num_pages=40,
        cfg=document_reader.DocumentReaderConfig(enable_ocr=False),
    )

    assert quality.status == "NEEDS_OCR"
    assert quality.text_pages_ge10 == 5
    assert quality.text_page_share_ge10 == 0.125
    assert len(quality.page_quality) == 40


def test_merge_page_texts_keeps_richest_text_per_page():
    preferred = [
        document_reader.PageText(1, "short text", "preferred"),
        document_reader.PageText(2, "image OCR text " * 20, "preferred"),
    ]
    native = [
        document_reader.PageText(1, "native annual report text " * 20, "native"),
        document_reader.PageText(2, "tiny", "native"),
    ]

    pages, backend = document_reader.merge_page_texts(
        [("preferred", preferred), ("native", native)]
    )

    assert pages[0].backend == "native"
    assert pages[1].backend == "preferred"
    assert backend == "pagewise_best[preferred,native]"


def test_run_ocrmypdf_force_mode_handles_signed_working_copy(tmp_path, monkeypatch):
    input_pdf = tmp_path / "signed_report.pdf"
    input_pdf.write_bytes(b"%PDF-1.4 signed bytes")
    output_dir = tmp_path / "ocr"
    calls = []

    def fake_run(cmd, capture_output, text, check):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"%PDF-1.4 OCR bytes")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(document_reader.subprocess, "run", fake_run)
    output = document_reader.run_ocrmypdf(
        str(input_pdf),
        str(output_dir),
        mode="force",
        invalidate_digital_signatures=True,
    )

    assert Path(output).exists()
    assert "--force-ocr" in calls[0]
    assert "--invalidate-digital-signatures" in calls[0]
