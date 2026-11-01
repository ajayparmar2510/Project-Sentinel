from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
OUT = ROOT / "explainability"

OUT.mkdir(parents=True, exist_ok=True)

TEST_FILE = RESULTS / "phase16_test_semantic_model.csv"
SAMPLE_SIZE = 5000
REPEATS = 3
RANDOM_STATE = 42

print("Phase 5: Permutation Importance")
print("Project:", ROOT)
print("Test file:", TEST_FILE)

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

df = pd.read_csv(TEST_FILE)
if "fake_label" not in df.columns:
    raise ValueError("Target column fake_label is missing.")

y = df["fake_label"].astype(int)

if y.nunique() != 2:
    raise ValueError("Expected both genuine and fake labels in test data.")

print(f"Test records: {len(df):,}")
print(f"Fake reviews: {(y == 1).sum():,}")
print(f"Genuine reviews: {(y == 0).sum():,}")

sample_size = min(SAMPLE_SIZE, len(df))

all_results = []
summary = {
    "method": "Permutation importance",
    "scoring_metric": "Average precision (PR-AUC)",
    "sample_size": sample_size,
    "repeats_per_feature": REPEATS,
    "random_state": RANDOM_STATE,
    "interpretation": (
        "Importance is the decrease in average precision after shuffling "
        "a feature. It measures predictive reliance, not causation."
    ),
    "models": {}
}

for model_name in ["baseline", "semantic"]:
    print("\n" + "=" * 60)
    print(f"MODEL: {model_name.upper()}")
    print("=" * 60)

    model_path = MODELS / f"phase16_{model_name}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    model = joblib.load(model_path)
    feature_names = list(model.feature_names_in_)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = df[feature_names].copy()

    if len(feature_names) != model.n_features_in_:
        raise ValueError("Model feature count mismatch.")

    # Use a fixed, stratified sample of the held-out test set.
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    baseline_ap = average_precision_score(
        y_sample,
        model.predict_proba(X_sample)[:, 1],
    )

    print("Features:", len(feature_names))
    print("Sample records:", len(X_sample))
    print(f"Sample average precision: {baseline_ap:.6f}")
    print("Calculating permutation importance; this may take a few minutes...")

    start = time.time()
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="average_precision",
        n_repeats=REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    elapsed = time.time() - start

    importance_df = pd.DataFrame({
        "model": model_name,
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    csv_path = OUT / f"phase5_{model_name}_permutation_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    all_results.append(importance_df)

    top = importance_df.head(15).sort_values("importance_mean")

    plt.figure(figsize=(10, 7))
    plt.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel("Decrease in average precision after shuffling")
    plt.ylabel("Feature")
    plt.title(
        f"Top 15 Permutation Importances — {model_name.title()} Model\n"
        f"Stratified test subset: {len(X_sample):,} records"
    )
    plt.tight_layout()

    plot_path = OUT / f"phase5_{model_name}_permutation_importance.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()

    summary["models"][model_name] = {
        "feature_count": len(feature_names),
        "sample_records": len(X_sample),
        "sample_average_precision": float(baseline_ap),
        "elapsed_seconds": round(elapsed, 2),
        "csv_file": csv_path.name,
        "plot_file": plot_path.name,
        "top_10_features": [
            {
                "feature": row.feature,
                "importance_mean": float(row.importance_mean),
                "importance_std": float(row.importance_std),
            }
            for row in importance_df.head(10).itertuples(index=False)
        ],
    }

    print(f"Finished in {elapsed:.1f} seconds.")
    print("\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))
    print("Saved:", csv_path)
    print("Saved:", plot_path)

combined = pd.concat(all_results, ignore_index=True)
combined_path = OUT / "phase5_all_models_permutation_importance.csv"
combined.to_csv(combined_path, index=False)

summary_path = OUT / "phase5_permutation_importance_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n" + "=" * 60)
print("PHASE 5 PERMUTATION IMPORTANCE COMPLETE")
print("=" * 60)
print("Combined results:", combined_path)
print("Summary:", summary_path)
print("All outputs saved in:", OUT)
print("No models were retrained or modified.")
