import json
import os
import subprocess
import sys

from fintech_adoption.config import resolve_project_root


def test_offline_demo_completes_under_windows_encoding(tmp_path):
    root = resolve_project_root()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1258"
    result = subprocess.run(
        [sys.executable, "-m", "fintech_adoption.demo", "--output", str(tmp_path)],
        cwd=root, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads((tmp_path / "DEMO_BANK_2023_demo_result.json").read_text(encoding="utf-8"))
    assert output["synthetic"] is True
    assert output["embedding_mode"] == "hashing"
    assert output["record"]["n_segments_adopted"] > 0
    assert output["record"]["n_segments_discussed"] > 0
