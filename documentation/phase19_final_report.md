# Project Sentinel — Phase 19: Error Analysis and Validation

## Objective
Evaluate baseline and semantic-enhanced fake-review classifiers on the held-out temporal test set, inspect prediction changes, and compare performance across rating and review-length groups.

## Overall test results

| Metric | Baseline | Semantic model |
|---|---:|---:|
| Test reviews | 91,717 | 91,717 |
| Decision threshold | 0.20 | 0.20 |
| Accuracy | 0.8065 | 0.8047 |
| Precision | 0.3421 | 0.3419 |
| Recall | 0.5028 | 0.5168 |
| F1 | 0.4071 | 0.4116 |
| ROC-AUC | 0.7934 | 0.7950 |
| PR-AUC | 0.3729 | 0.3764 |
| True positives | 6,094 | 6,264 |
| False positives | 11,721 | 12,056 |
| False negatives | 6,026 | 5,856 |

The semantic model increased recall by approximately 1.40 percentage points and F1 by 0.0044. It identified 170 additional fake-labeled reviews, while generating 335 additional false positives. Accuracy decreased by approximately 0.18 percentage points.

These differences are modest and have not been established as statistically significant.

## Prediction-change analysis

A total of 3,693 of 91,717 test predictions changed between models, a rate of 4.027%.

| Rating | Test reviews | Changed predictions | Change rate |
|---|---:|---:|---:|
| 1 star | 6,420 | 134 | 2.087% |
| 2 stars | 6,438 | 394 | 6.120% |
| 3 stars | 11,123 | 360 | 3.237% |
| 4 stars | 29,431 | 823 | 2.796% |
| 5 stars | 38,305 | 1,982 | 5.174% |

| Review length | Test reviews | Changed predictions | Change rate |
|---|---:|---:|---:|
| 0–20 words | 8,310 | 626 | 7.533% |
| 21–50 words | 23,421 | 1,309 | 5.589% |
| 51–100 words | 26,830 | 1,039 | 3.873% |
| 101–200 words | 22,063 | 509 | 2.307% |
| 201+ words | 11,093 | 210 | 1.893% |

The largest change rate occurred in reviews of 0–20 words. Two-star reviews had the highest rating-specific change rate. Prediction changes are not necessarily corrections; they can introduce or remove errors.

## Subgroup performance

### By rating

| Rating | Baseline F1 | Semantic F1 | Difference |
|---|---:|---:|---:|
| 1 star | 0.5336 | 0.5303 | -0.0033 |
| 2 stars | 0.3975 | 0.3918 | -0.0057 |
| 3 stars | 0.3467 | 0.3464 | -0.0002 |
| 4 stars | 0.3598 | 0.3665 | +0.0068 |
| 5 stars | 0.3861 | 0.3959 | +0.0098 |

F1 improved for 2 of 5 rating groups and worsened for 3 of 5. The largest improvement was in five-star reviews; the largest decline was in two-star reviews.

### By review length

| Review length | Baseline F1 | Semantic F1 | Difference |
|---|---:|---:|---:|
| 0–20 words | 0.5038 | 0.5048 | +0.0010 |
| 21–50 words | 0.3811 | 0.3869 | +0.0058 |
| 51–100 words | 0.3685 | 0.3752 | +0.0067 |
| 101–200 words | 0.4086 | 0.4139 | +0.0053 |
| 201+ words | 0.4190 | 0.4137 | -0.0053 |

F1 improved for 4 of 5 review-length groups and worsened for 1 of 5. The strongest gain occurred in the 51–100-word group. The 201+-word group experienced lower recall and F1.

## Validation and limitations

- Test predictions were checked against the saved threshold of 0.20.
- Subgroup analyses used the same held-out test reviews for both models.
- The semantic-enhanced model adds two semantic features to the 41-feature baseline.
- YelpZip labels are the evaluation reference and are not independent confirmation of fraud.
- Subgroup differences are descriptive; no statistical significance is claimed.
- Model outputs must not be treated as definitive fraud verdicts.
- Accuracy alone is insufficient for evaluating this imbalanced classification task.
- The results support retaining semantic features for further evaluation, not unconditional replacement of the baseline.

## Phase 19 deliverables

- `results/phase19_overall_test_metrics.csv`
- `results/phase19_confusion_matrix_counts.csv`
- `results/phase19_subgroup_error_analysis.csv`
- `results/phase19_all_test_errors.csv`
- `results/phase19_representative_test_errors.csv`
- `results/phase19_error_analysis_summary.json`
- `results/phase19_changed_prediction_cases.csv`
- `results/phase19_changed_prediction_summary.csv`
- `results/phase19_changed_predictions_by_rating.csv`
- `results/phase19_changed_predictions_by_length.csv`
- `results/phase19_change_rates_by_rating.csv`
- `results/phase19_change_rates_by_review_length_group.csv`
- `results/phase19_subgroup_performance_by_rating.csv`
- `results/phase19_subgroup_performance_by_length.csv`
- `results/phase19_subgroup_performance_summary.json`

## Conclusion

Phase 19's overall error analysis, changed-prediction inspection, change-rate calculations, and subgroup performance comparison have been completed. The semantic model provides a modest overall improvement in recall and F1, with mixed effects across subgroups. Further validation is needed before making stronger claims about generalization or practical fraud detection.
