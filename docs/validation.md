# Validation

Validated on Windows with Python 3.12.10 on 2026-11-08.

| Check | Result |
| --- | --- |
| Fresh environment installation | Passed |
| Dependency consistency (`pip check`) | Passed |
| Ruff (`E9`, `F`) | Passed |
| Synthetic and mocked regression tests | 87 passed |
| Pipeline CLI help | Passed |
| Offline demo under Windows text encoding | Passed |
| Source distribution and wheel build | Passed |
| Installed wheel run outside the repository | Passed |
| Release source hashes and credential/path scan | Passed |

The offline example contains 10 synthetic segments. Its mock classifier marks 6 adopted, 2 discussed and 2 irrelevant. These counts are a smoke check, not measured accuracy.

The test environment uses NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1 and scikit-learn 1.9.1. Exact versions are in `requirements/tested-windows-py312.txt`. That snapshot applies to the Windows/Python 3.12 test environment. Optional PDF parsers, semantic embedding models, OCR and live provider calls need separate validation.

The GitHub workflow defines Python 3.10 and 3.12 jobs on Windows and Ubuntu. Its live results are available in the repository's Actions tab.
