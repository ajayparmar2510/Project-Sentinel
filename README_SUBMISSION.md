# Project Sentinel
## Explainable Trust Intelligence Framework for Fake Review Detection

### 1. Project overview

Project Sentinel explores fake-review detection by combining engineered review,
reviewer, product, and temporal features with semantic similarity features.
It also develops exploratory reviewer and product risk-prioritization profiles.

The project is an experimental research framework. A predicted risk score or
flag is not proof that a review or reviewer is fraudulent.

### 2. Dataset and evaluation design

- Dataset: YelpZip, reconstructed into `yelpzip_final.csv`.
- Total records: 608,598.
- Fake-labeled records: 80,466 (approximately 13.22%).
- Genuine-labeled records: 528,132 (approximately 86.78%).
- Evaluation: chronological train, validation, and held-out test partitions.
- Train: 425,794 records.
- Validation: 91,087 records.
- Test: 91,717 records.

The test partition was used for final evaluation rather than threshold tuning.
YelpZip labels are the evaluation reference, not independent confirmation of
fraud.

### 3. Model comparison

The baseline uses 41 engineered features. The semantic model adds two features:
maximum cosine similarity to up to 20 reviews of the same product from strictly
earlier dates, and the count of those prior product reviews.

Both models use the validation-selected threshold of 0.20 for this comparison.

| Test metric | Baseline | Semantic model |
|---|---:|---:|
| Accuracy | 0.8065 | 0.8047 |
| Precision | 0.3421 | 0.3419 |
| Recall | 0.5028 | 0.5168 |
| F1 | 0.4071 | 0.4116 |
| ROC-AUC | 0.7934 | 0.7950 |
| PR-AUC | 0.3729 | 0.3764 |

The semantic model shows modest gains in recall, F1, ROC-AUC, and PR-AUC, with
a small decrease in accuracy and a slight decrease in precision. The results
do not establish statistical significance or generalization beyond this
dataset.

### 4. Trust Intelligence Score

The trust/risk scoring stage evaluates heuristic trust signals and a
regularized logistic-regression candidate. The exported score is a relative
risk index for exploratory prioritization, not a calibrated probability of
fraud and not a definitive reviewer verdict.

Reviewer and product profiles summarize score distributions and review
activity. They must not be interpreted as proof of wrongdoing.

### 5. Error analysis

Phase 19 evaluates the baseline and semantic models on the same held-out test
records, examines confusion matrices, reviews changed predictions, and
compares performance by rating and review length.

The semantic model changes approximately 4.03% of test predictions. Subgroup
effects are mixed: F1 improves in four of five review-length groups and two of
five rating groups. These descriptive comparisons have not been shown to be
statistically significant.

### 6. Main artifact locations

- `results/`: metrics, audits, score exports, profile tables, and error analyses.
- `models/`: serialized baseline, semantic, and trust-scoring models.
- `scripts/`: scripts used to generate and audit results.
- `documentation/`: phase reports and the submission manifest.
- `explainability/`: explainability-related outputs, when present.
- `logs/`: captured execution logs, when present.

See `documentation/phase19_final_report.md` for the detailed error analysis
and `documentation/phase20_submission_manifest.json` for the artifact
inventory and SHA-256 hashes.

### 7. Reproducibility

Use the project's Python virtual environment and the scripts documented in
the project workflow. Dataset paths in the scripts may be specific to the
original macOS working directory. Check dependencies and paths before
re-running expensive embedding or model-training stages.

The saved manifest records file hashes at the time of generation. If any
artifact is changed later, regenerate the manifest.

### 8. Limitations and responsible use

- YelpZip labels are imperfect proxies for the task's target.
- The evaluation is dataset-specific and does not establish real-world fraud
  detection effectiveness.
- Accuracy alone is insufficient for an imbalanced classification problem.
- False positives can wrongly flag genuine reviews.
- Trust scores and profile rankings should be used, if at all, as review
  prioritization aids requiring human review, not as automatic penalties.
- No claim of statistical significance is made.
- Further validation on independent data and calibration analysis would be
  required before practical deployment.

### 9. Current status

Phases 16–19 have generated semantic features, model comparisons, exploratory
trust scoring, reviewer/product profiles, and subgroup error analyses.
Phase 20 integration is underway; this README and the artifact manifest are
submission-preparation documents, not certification of external validity.
