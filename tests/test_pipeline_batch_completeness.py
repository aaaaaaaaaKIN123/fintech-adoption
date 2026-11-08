import pandas as pd
import pytest

import fintech_adoption.pipeline as pipeline
from fintech_adoption.config import PipelineConfig


class FakeEmbedder:
    def __init__(self, *args, **kwargs):
        pass


def test_run_batch_raises_and_logs_when_preprocess_drops_report(tmp_path, monkeypatch):
    cfg = PipelineConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.log_dir = str(tmp_path / "logs")

    monkeypatch.setattr(pipeline, "FinTechEmbedder", FakeEmbedder)
    monkeypatch.setattr(pipeline, "phase0_init_seeds", lambda cfg, embedder: None)

    def fake_preprocess_batch(*args, failures=None, **kwargs):
        if failures is not None:
            failures.append(
                {
                    "bank_id": "ACB",
                    "year": 2020,
                    "pdf_path": "data/raw/acb.pdf",
                    "stage": "preprocess",
                    "error": "PDF read failed",
                }
            )
        return []

    monkeypatch.setattr(pipeline, "preprocess_batch", fake_preprocess_batch)

    with pytest.raises(pipeline.BatchCompletenessError) as excinfo:
        pipeline.run_batch({("ACB", 2020): "data/raw/acb.pdf"}, cfg, mock_llm=True)

    failure_log = tmp_path / "logs" / "fta_failed_reports.csv"
    assert "ACB 2020" in str(excinfo.value)
    assert failure_log.exists()
    failures = pd.read_csv(failure_log)
    assert failures.loc[0, "bank_id"] == "ACB"
    assert failures.loc[0, "stage"] == "preprocess"
