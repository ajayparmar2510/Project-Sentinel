# Phase 5 — Explainability and Feature Importance

## 1. Objective

This phase examines which input features the trained fake-review detection
models rely on most when distinguishing reviews labelled fake from those
labelled genuine.

## 2. Models evaluated

- Baseline HistGradientBoostingClassifier: 41 features.
- Semantic HistGradientBoostingClassifier: 43 features.

Both saved models were loaded from the existing project artifacts. Neither
model was retrained or modified.

## 3. Methodology

Permutation importance was calculated using scikit-learn with average
precision as the scoring metric. Average precision summarizes precision
and recall across score thresholds and is useful for an imbalanced
classification problem.

The evaluation used a fixed, stratified sample of 5,000 records from the
saved held-out test dataset, with three permutations per feature and
random state 42.

Importance is the decrease in average precision after shuffling a feature.
Higher positive values indicate greater predictive reliance in this
evaluation. Small or negative values can occur because of feature overlap,
sampling variability, or limited predictive contribution.

## 4. Baseline model: top 10 features

| Feature | Mean permutation importance |
|---|---:|
| days_since_product_previous_review | 0.052309 |
| user_prior_avg_review_length | 0.039622 |
| user_prior_reviews | 0.036240 |
| days_since_user_previous_review | 0.029197 |
| punctuation_count | 0.025879 |
| product_prior_reviews_30d | 0.024505 |
| product_rating_deviation | 0.022588 |
| product_prior_avg_rating | 0.019473 |
| product_prior_reviews | 0.017485 |
| product_prior_rating_variance | 0.012986 |

## 5. Semantic model: top 10 features

| Feature | Mean permutation importance |
|---|---:|
| days_since_product_previous_review | 0.055510 |
| user_prior_reviews | 0.045549 |
| user_prior_avg_review_length | 0.042298 |
| days_since_user_previous_review | 0.029229 |
| product_prior_reviews_30d | 0.027711 |
| punctuation_count | 0.027506 |
| product_prior_avg_rating | 0.021737 |
| product_rating_deviation | 0.021192 |
| product_prior_rating_variance | 0.014812 |
| product_prior_reviews | 0.011728 |

## 6. Semantic feature analysis

The two additional semantic-model features had the following measured
importance on the sampled test records:

| Feature | Rank | Mean importance |
|---|---:|---:|
| prior_product_review_count | 12 of 43 | 0.008386 |
| max_prior_20_product_similarity | 26 of 43 | 0.001873 |

The prior-product-review count had a stronger measured contribution than
the maximum similarity feature in this experiment. Similarity ranked in
the lower half of the feature list.

Therefore, the results do not support claiming that text similarity alone
accounts for the semantic model's improvement. The semantic model adds
both historical review-count information and similarity information.

## 7. Sample-level average precision

- Baseline model: 0.373503.
- Semantic model: 0.380480.
- Difference on this sample: +0.006977.

This is a descriptive comparison on the same 5,000-record sample. It must
not be presented as a replacement for the previously recorded full
held-out test metrics.

## 8. Interpretation

The most important features in this analysis largely concern review
timing, reviewer history, and product activity. This suggests that
temporal and behavioural patterns provide useful predictive signals in
the trained models.

Permutation importance measures predictive reliance, not causation.
A high score does not prove that a review or reviewer is fraudulent.
Importance can also be affected by correlated or overlapping features.

The analysis is sample-based, uses three permutations per feature, and
should be treated as exploratory evidence rather than a definitive
ranking. The model's predictions and risk scores require human review
and should not be used alone to accuse or penalize a reviewer.

## 9. Output artifacts

- phase5_baseline_permutation_importance.csv
- phase5_baseline_permutation_importance.png
- phase5_semantic_permutation_importance.csv
- phase5_semantic_permutation_importance.png
- phase5_all_models_permutation_importance.csv
- phase5_permutation_importance_summary.json
- phase5_explainability_report.md

## 10. Conclusion

Phase 5 establishes an initial, reproducible feature-importance analysis
for both trained models. Historical activity and timing features rank
highly. The semantic model's prior-product-review count contributes more
strongly than its maximum prior-review similarity feature in the sampled
evaluation. These findings support a cautious, evidence-based explanation
of model behaviour without implying causal relationships or certainty
about individual reviews.
