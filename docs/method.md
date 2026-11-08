# Method and limitations

The code combines a fixed bilingual concept taxonomy with sentence-transformer retrieval, paragraph coherence, phrase-level TF-IDF and adoption-verb features. A provider returns structured labels for the selected passages. The weighted hybrid measure and supervised logistic stacking model are separate estimators.

`ADOPTED` requires a bank acting on a specific capability with evidence of implementation. `DISCUSSED_ONLY` and `IRRELEVANT` receive zero in the hybrid score. Invalid JSON, non-boolean adoption flags and non-finite probabilities fail closed.

The segment score is `beta1 * S1 + beta2 * S2 + beta3 * P_llm` for adopted passages. The report measure divides their sum by `max(canonical_clean_words / 10000, 1)`. The floor matters for short or incompletely extracted reports. Scores describe the disclosed evidence in the available text; they are not direct measurements of technology expenditure or implementation quality.

Top-K retrieval can omit relevant evidence. Bilingual deduplication is heuristic and needs manual checks. Extraction quality checks cannot guarantee that every relevant passage was recovered. LLM labels can be wrong and provider/model updates can change them. The mock classifier is a simple smoke-test tool.

Stacking needs completed human labels. Blank or unknown labels, non-binary numeric targets and duplicate annotation keys are rejected. Grouped validation separates bank-year observations, but the same bank can appear in both partitions in different years. A new-bank generalization study needs a bank-level split. Small or highly imbalanced grouped samples may not support held-out validation.

Do not treat plausible time trends or bank rankings as proof of validity. Assess extraction completeness, recall on independently sampled passages, annotator agreement, precision/recall on held-out human labels, model sensitivity and stability across banks and years. No empirical accuracy benchmark is claimed by this release.

Retain report hashes, configuration, package/dependency versions, seed definitions, prompt/model identity, timestamps and the JSONL audit trail for each research run. Preserve any AI assistance disclosure required by the research venue.
