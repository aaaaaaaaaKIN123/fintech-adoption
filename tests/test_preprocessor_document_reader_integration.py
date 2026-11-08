import fintech_adoption.preprocessor as preprocessor
from fintech_adoption.document_reader import DocumentReadResult, PageText


def test_preprocess_report_uses_document_reader_metadata(tmp_path, monkeypatch):
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    def fake_reader(path, cfg=None):
        return DocumentReadResult(
            pdf_path=str(pdf_path),
            backend="ocrmypdf+pymupdf4llm",
            pages=[
                PageText(
                    1,
                    "The bank launched eKYC for customers in 2023 with mobile banking adoption.",
                    "pymupdf4llm",
                )
            ],
            status="OK",
            ocr_used=True,
            ocr_engine="ocrmypdf",
            ocr_pdf_path="data/processed/ocr/report_ocr.pdf",
            scan_detected=True,
            text_coverage=0.71,
            raw_total_words=12,
            num_pages=1,
            extraction_warnings=["example warning"],
        )

    monkeypatch.setattr(preprocessor, "read_pdf_auto", fake_reader)

    report = preprocessor.preprocess_report(
        str(pdf_path),
        "VCB",
        2023,
        min_words=5,
        max_words=100,
    )

    assert report.metadata["pdf_backend"] == "ocrmypdf+pymupdf4llm"
    assert report.metadata["ocr_used"] is True
    assert report.metadata["scan_detected"] is True
    assert report.metadata["text_coverage"] == 0.71
    assert report.metadata["extraction_status"] == "OK"
    assert report.raw_total_words == 12
