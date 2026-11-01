from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

OUT = Path("results")
PRED = OUT / "phase16_model_comparison_test_predictions.csv"
SOURCE = Path("../..") / "yelpzip_final.csv"

pred = pd.read_csv(PRED)
source = pd.read_csv(
    SOURCE,
    usecols=["rating", "review_text"],
)
source.insert(0, "row_id", np.arange(1, len(source) + 1))

df = pred.merge(
    source.assign(
        review_word_count=(
            source["review_text"].fillna("").astype(str).str.split().str.len()
        )
    )[["row_id", "rating", "review_word_count"]],
    on="row_id",
    how="left",
    validate="one_to_one",
)

assert len(df) == 91717, f"Unexpected test row count: {len(df)}"
assert df["row_id"].is_unique, "Duplicate row IDs"
assert df[["rating", "review_word_count"]].notna().all().all(), (
    "Missing subgroup metadata"
)

for model in ["baseline", "semantic"]:
    expected = (df[f"{model}_probability"] >= 0.20).astype(int)
    assert (expected == df[f"{model}_prediction"]).all(), (
        f"{model} predictions do not match threshold 0.20"
    )

df["review_length_group"] = pd.cut(
    df["review_word_count"],
    bins=[-1, 20, 50, 100, 200, float("inf")],
    labels=["0-20", "21-50", "51-100", "101-200", "201+"],
)

def calculate_metrics(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
    }

summary = {
    "phase": "19.5",
    "test_rows": len(df),
    "threshold": 0.20,
    "groups": {},
    "caveat": (
        "Descriptive subgroup evaluation only. YelpZip labels are the "
        "evaluation reference, not independent confirmation of fraud. "
        "No statistical significance is claimed."
    ),
}

for column, name in [
    ("rating", "rating"),
    ("review_length_group", "length"),
]:
    rows = []

    for group, part in df.groupby(column, observed=True, dropna=False):
        y = part["fake_label"].astype(int).to_numpy()
        baseline = part["baseline_prediction"].astype(int).to_numpy()
        semantic = part["semantic_prediction"].astype(int).to_numpy()

        bm = calculate_metrics(y, baseline)
        sm = calculate_metrics(y, semantic)

        row = {
            "group": str(group),
            "test_reviews": len(part),
            "fake_prevalence_pct": 100 * y.mean(),
        }

        for metric in ["accuracy", "precision", "recall", "f1"]:
            row[f"baseline_{metric}"] = bm[metric]
            row[f"semantic_{metric}"] = sm[metric]
            row[f"semantic_minus_baseline_{metric}"] = (
                sm[metric] - bm[metric]
            )

        rows.append(row)

    result = pd.DataFrame(rows)
    path = OUT / f"phase19_subgroup_performance_by_{name}.csv"
    result.to_csv(path, index=False)

    display = result.copy()
    for col in display.columns:
        if col.endswith("_pct"):
            display[col] = display[col].round(2)
        elif any(
            col.endswith("_" + metric)
            for metric in ["accuracy", "precision", "recall", "f1"]
        ):
            display[col] = display[col].round(4)

    print(f"\n=== SUBGROUP PERFORMANCE: {name.upper()} ===")
    print(display.to_string(index=False))
    print(f"Saved: {path}")

    summary["groups"][name] = {
        "group_count": len(result),
        "f1_improved": int(
            (result["semantic_minus_baseline_f1"] > 0).sum()
        ),
        "f1_worsened": int(
            (result["semantic_minus_baseline_f1"] < 0).sum()
        ),
        "f1_unchanged": int(
            (result["semantic_minus_baseline_f1"] == 0).sum()
        ),
    }

summary_path = OUT / "phase19_subgroup_performance_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print("\n=== SUBGROUP PERFORMANCE SUMMARY ===")
print(json.dumps(summary["groups"], indent=2))
print(f"Saved: {summary_path}")
print("\nPHASE 19.5 SUBGROUP PERFORMANCE AUDIT COMPLETE")
