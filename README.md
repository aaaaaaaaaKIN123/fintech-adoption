# FinTech adoption

Python tools for measuring evidence of FinTech adoption in Vietnamese and English bank annual reports. The pipeline retrieves candidate passages, scores phrases, asks an LLM to distinguish deployed capabilities from plans, and exports bank-year measures with passage-level evidence.

This repository contains the reusable measurement code from a larger research workspace. Bank reports, research datasets, manuscripts and fitted models are excluded.

## Try it

Use Python 3.10 or newer. From the repository root:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
fta-demo
```

The demo runs offline with synthetic Vietnamese text, hashed text features and a deterministic mock classifier. It writes an explicitly marked example to `data/analysis_output/DEMO_BANK_2023_demo_result.json`. Its scores demonstrate the workflow and have no empirical interpretation.

For sentence-transformer embeddings instead:

```sh
python -m pip install -e ".[embeddings]"
fta-demo --semantic
```

The first semantic run downloads model weights. `--mock_llm` replaces only the LLM stage; a PDF run still needs embeddings and PDF dependencies.

## Process reports

```sh
python -m pip install -e ".[pdf,embeddings,openai]"
```

1. Put annual-report PDFs in `data/raw/`.
2. Copy `examples/report_inventory.csv` to `data/metadata/report_inventory.csv` and replace the example row with your reports.
3. Create a sample CSV using `examples/bank_sample.csv` as the schema. Set the included banks and years for your own study.
4. Run the PDF quality check, then the panel builder:

```sh
python data/scripts/04_check_report_pdf_quality.py --sample examples/bank_sample.csv --start-year 2023 --end-year 2023
python data/scripts/01_build_fta_panel.py --mock-llm
```

For live adjudication, set `OPENAI_API_KEY` in your shell, omit `--mock-llm` and supply `--llm-model` if you need a different model. Live runs send candidate passages to the configured provider and incur API usage charges. The builder checks connectivity before extraction. The optional Anthropic provider needs the `anthropic` extra and `ANTHROPIC_API_KEY`.

Scanned reports also need OCRmyPDF, Tesseract with Vietnamese/English language data, and Ghostscript installed separately. Add `--enable-ocr` to the quality check. OCR copies go to `data/processed/ocr/`.

## What it measures

- Keyword frequency and entropy-weighted indices provide disclosure baselines.
- `hybrid_FTA_it` counts weighted evidence only when adjudication returns `ADOPTED`. Plans and general discussion contribute zero.
- An optional logistic stacking model learns from human labels and validates by bank-year groups.

The denominator uses canonical clean report words after bilingual deduplication. The implementation floors the denominator at one 10,000-word unit for short reports. Read [the method and limitations](docs/method.md) before interpreting scores.

## Outputs

| Path | Contents |
| --- | --- |
| `data/analysis_output/FTA_panel.csv` | Bank-year measures and diagnostics |
| `data/analysis_output/FTA_baseline_indices.csv` | Keyword and entropy baselines |
| `data/analysis_output/FTA_segment_features.csv` | Features for annotation and stacking |
| `data/analysis_output/*_segments.jsonl` | Passage text, page, labels and evidence |
| `results/logs/` | Extraction quality and failures |

Defaults are relative to the current working directory. Set `FTA_WORKSPACE` to a different workspace directory when calling the installed package from elsewhere. Run the research scripts from this repository root; they resolve paths relative to their own location.

## Development

```sh
python -m pytest -q
python -m ruff check .
python -m build
```

Tests use synthetic inputs and mocked provider/PDF calls. They do not validate the accuracy of adoption estimates on real reports. See [CONTRIBUTING.md](CONTRIBUTING.md) and [the release review](docs/release-review.md) for scope and remaining work.

The [validation record](docs/validation.md) lists the local checks and tested dependency versions.

## Reuse

No software license has been selected for this release. Ask the repository owner before redistributing or reusing the code. Report sources and model weights have their own terms. When describing research that uses this method, specify the seed list, model versions, prompts, thresholds, annotation protocol and input-report versions.
