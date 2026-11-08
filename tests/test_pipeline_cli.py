import os
import subprocess
import sys

from fintech_adoption.config import resolve_project_root


def test_pipeline_help_is_safe_under_windows_cp1258_encoding():
    root = resolve_project_root()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1258"

    completed = subprocess.run(
        [sys.executable, "-m", "fintech_adoption.pipeline", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--alpha" in completed.stdout
