# Contributing

Install with `python -m pip install -e ".[dev]"`, then run `python -m pytest -q` and `python -m ruff check .`.

Describe the behavior being changed and include a small synthetic example for scoring or extraction bugs. Add a regression test when a change affects measurement. Keep raw reports and provider credentials out of commits. Do not use live API calls in the test suite.

Changes to seeds, prompts, thresholds, deduplication or denominators can alter research results. Document those changes and regenerate affected outputs in your own workspace. Record AI assistance where required by your institution or publication venue.
