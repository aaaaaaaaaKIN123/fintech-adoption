# Release review

Version 0.1.0 packages the measurement pipeline. Its useful parts are bilingual report preprocessing, explicit adoption/discussion labels, evidence exports, disclosure baselines and optional supervised stacking. These are engineering contributions; this release makes no claim that their combination is a new research method.

The original workspace had no Git repository or installable Python package. Some instructions referred to a directory that did not exist, and OpenAI was missing from the default dependency instructions. The demo could match diagnostic field names instead of passage text and label every passage as discussion.

This release adds a package namespace, command-line entry points, optional dependency groups, an offline synthetic demo and CI. It removes decorative comment banners and unused imports while retaining the equations, prompts, evidence fields and research safeguards. The original research manuscripts and their disclosures are outside the release.

Fixes cover malformed structured responses, string-valued booleans, non-finite probabilities, empty embedding matrices with non-default dimensions, mock classification contaminated by prompt metadata and invalid human annotations being silently treated as negative examples. Regression tests exercise these cases.

Remaining work:

- Review annotator agreement and held-out label quality before training.
- Benchmark retrieval and adjudication with independent human annotations.
- Record a dependency lock for a chosen full PDF/embedding environment. The lightweight test environment is recorded separately.
- Test OCR and live provider calls on a licensed corpus. Unit tests mock those boundaries.
- Review data redistribution rights and choose a code license before encouraging reuse.

Version-specific econometric models, manuscript builders and report-download automation remain in the research workspace. They need a separate data and portability review before release.
