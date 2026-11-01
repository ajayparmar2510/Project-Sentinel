import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

M3_DIR = "/Users/apple/Downloads/YelpZip_Milestone2/Milestone3"
M4_DIR = os.path.join(M3_DIR, "Milestone4")
RESULTS_DIR = os.path.join(M4_DIR, "results")
MODELS_DIR = os.path.join(M4_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = "fake_label"
SEMANTIC_COLUMNS = [
    "max_prior_20_product_similarity",
    "prior_product_review_count",
]
BASE_COLUMNS = {
    "row_id", "user_id", "prod_id", "rating", "fake_label", "date"
}
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

SPLITS = {
    "train": (
        os.path.join(M3_DIR, "results", "train_model.csv"),
        os.path.join(RESULTS_DIR, "phase16_train_semantic_model.csv"),
    ),
    "validation": (
        os.path.join(M3_DIR, "results", "validation_model.csv"),
        os.path.join(RESULTS_DIR, "phase16_validation_semantic_model.csv"),
    ),
    "test": (
        os.path.join(M3_DIR, "results", "test_model.csv"),
        os.path.join(RESULTS_DIR, "phase16_test_semantic_model.csv"),
    ),
}

MODEL_PARAMETERS = {
    "learning_rate": 0.08,
    "max_iter": 250,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 42,
}

def make_model():
    """Create the same HistGradientBoostingClassifier used in M3."""
    return HistGradientBoostingClassifier(
        learning_rate=MODEL_PARAMETERS["learning_rate"],
        max_iter=MODEL_PARAMETERS["max_iter"],
        max_leaf_nodes=MODEL_PARAMETERS["max_leaf_nodes"],
        min_samples_leaf=MODEL_PARAMETERS["min_samples_leaf"],
        l2_regularization=MODEL_PARAMETERS["l2_regularization"],
        random_state=MODEL_PARAMETERS["random_state"],
    )


def load_datasets():
    data = {}
    for split, (base_path, semantic_path) in SPLITS.items():
        if not os.path.isfile(base_path):
            raise FileNotFoundError(f"Missing M3 dataset: {base_path}")
        if not os.path.isfile(semantic_path):
            raise FileNotFoundError(f"Missing M4 dataset: {semantic_path}")

        print(f"Loading {split} datasets...")
        data[split] = {
            "base": pd.read_csv(base_path),
            "semantic": pd.read_csv(semantic_path),
        }
    return data

def audit_datasets(data):
    """Audit model datasets using the separate, row-aligned M3 audit files."""
    audit_paths = {
        "train": os.path.join(M3_DIR, "results", "train_audit.csv"),
        "validation": os.path.join(M3_DIR, "results", "validation_audit.csv"),
        "test": os.path.join(M3_DIR, "results", "test_audit.csv"),
    }

    required_audit_columns = {"row_id", "user_id", "prod_id", "date"}
    report = {
        "status": "passed",
        "checks": [],
        "splits": {},
        "notes": [
            "Model CSVs do not contain row_id or identity columns.",
            "Separate M3 audit CSVs provide row identity and ordering."
        ],
    }

    for split, pair in data.items():
        base = pair["base"]
        semantic = pair["semantic"]

        audit_path = audit_paths[split]
        audit = pd.read_csv(audit_path)

        if not required_audit_columns.issubset(audit.columns):
            raise ValueError(
                f"{split}: audit file is missing required columns: "
                f"{sorted(required_audit_columns - set(audit.columns))}"
            )

        if len(base) != len(semantic) or len(base) != len(audit):
            raise ValueError(
                f"{split}: row-count mismatch: "
                f"base={len(base)}, semantic={len(semantic)}, "
                f"audit={len(audit)}"
            )

        if audit["row_id"].isna().any() or audit["row_id"].duplicated().any():
            raise ValueError(f"{split}: row_id is missing or duplicated.")

        if audit[["user_id", "prod_id", "date"]].isna().any().any():
            raise ValueError(f"{split}: missing identity/date values in audit.")

        if TARGET not in base.columns or TARGET not in semantic.columns:
            raise ValueError(f"{split}: target column {TARGET!r} is missing.")

        if base.columns[-1] != TARGET or semantic.columns[-1] != TARGET:
            raise ValueError(f"{split}: target must be the last column.")

        base_features = [c for c in base.columns if c != TARGET]
        semantic_features = [c for c in semantic.columns if c != TARGET]

        expected_semantic_features = base_features + SEMANTIC_COLUMNS
        if semantic_features != expected_semantic_features:
            raise ValueError(
                f"{split}: feature columns/order do not match expectation. "
                f"Expected original features followed by {SEMANTIC_COLUMNS}."
            )

        if not base[TARGET].equals(semantic[TARGET]):
            raise ValueError(f"{split}: target labels differ between M3 and M4.")

        # Compare all original M3 features row-by-row without changing either file.
        try:
            pd.testing.assert_frame_equal(
                base[base_features],
                semantic[base_features],
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            raise ValueError(
                f"{split}: original M3 feature values/order changed in M4."
            ) from exc

        if base[base_features].isna().any().any():
            raise ValueError(f"{split}: missing values found in original features.")

        similarity = semantic["max_prior_20_product_similarity"]
        prior_count = semantic["prior_product_review_count"]

        # Similarity should be undefined exactly when there is no prior product history.
        expected_undefined = prior_count.eq(0)
        actual_undefined = similarity.isna()

        if not actual_undefined.equals(expected_undefined):
            mismatch_count = int((actual_undefined != expected_undefined).sum())
            raise ValueError(
                f"{split}: semantic missingness disagrees with prior counts "
                f"for {mismatch_count} rows."
            )

        defined = similarity.dropna()
        if not defined.empty and (
            (defined < -1.00001).any() or (defined > 1.00001).any()
        ):
            raise ValueError(f"{split}: cosine similarities fall outside [-1, 1].")

        report["splits"][split] = {
            "rows": int(len(base)),
            "audit_rows": int(len(audit)),
            "unique_row_ids": int(audit["row_id"].nunique()),
            "original_feature_count": int(len(base_features)),
            "semantic_feature_count": int(len(semantic_features)),
            "undefined_similarity_rows": int(actual_undefined.sum()),
            "defined_similarity_rows": int(similarity.notna().sum()),
            "zero_prior_product_history_rows": int(expected_undefined.sum()),
            "target_labels_identical": True,
            "original_features_identical_in_order": True,
        }

        report["checks"].append(f"{split}: PASS")

    report_path = os.path.join(
        RESULTS_DIR, "phase16_model_comparison_input_audit.json"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nAUDIT REPORT SAVED: {report_path}")
    return report



def validation_threshold_table(y_true, probabilities):
    """Evaluate candidate thresholds on validation data."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "accuracy": accuracy_score(y_true, predictions),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    return pd.DataFrame(rows)


def evaluate_at_threshold(y_true, probabilities, threshold):
    """Evaluate predictions at the selected decision threshold."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def run_experiment(data):
    results = []
    threshold_tables = []
    prediction_output = None

    for model_name in ["baseline", "semantic"]:
        print("\n" + "=" * 70)
        print(f"TRAINING: {model_name.upper()} MODEL")
        print("=" * 70)

        train_df = data["train"][model_name]
        val_df = data["validation"][model_name]
        test_df = data["test"][model_name]

        features = [
            c for c in train_df.columns if c not in BASE_COLUMNS
        ]
        val_features = [
            c for c in val_df.columns if c not in BASE_COLUMNS
        ]
        test_features = [
            c for c in test_df.columns if c not in BASE_COLUMNS
        ]

        if features != val_features or features != test_features:
            raise ValueError(f"{model_name}: feature order mismatch.")

        X_train = train_df[features]
        y_train = train_df[TARGET].astype(int)
        X_val = val_df[features]
        y_val = val_df[TARGET].astype(int)
        X_test = test_df[features]
        y_test = test_df[TARGET].astype(int)

        if model_name == "baseline":
            if X_train.isnull().any().any():
                raise ValueError("Baseline training features contain missing values.")
        else:
            # HistGradientBoostingClassifier supports NaN natively.
            # Preserve undefined similarity; do not impute it as zero.
            if X_train[features[:-2]].isnull().any().any():
                raise ValueError("Unexpected missing original features.")

        model = make_model()
        start = time.time()
        model.fit(X_train, y_train)
        training_seconds = time.time() - start

        val_probabilities = model.predict_proba(X_val)[:, 1]
        threshold_table = validation_threshold_table(y_val, val_probabilities)
        threshold_table.insert(0, "model", model_name)
        threshold_tables.append(threshold_table)

        # Same validation threshold grid and maximum-F1 selection rule.
        best_index = threshold_table["f1"].idxmax()
        selected_threshold = float(
            threshold_table.loc[best_index, "threshold"]
        )

        test_start = time.time()
        test_probabilities = model.predict_proba(X_test)[:, 1]
        prediction_seconds = time.time() - test_start

        metrics = evaluate_at_threshold(
            y_test, test_probabilities, selected_threshold
        )
        metrics.update({
            "model": model_name,
            "feature_count": len(features),
            "train_records": len(train_df),
            "validation_records": len(val_df),
            "test_records": len(test_df),
            "validation_best_f1": float(
                threshold_table.loc[best_index, "f1"]
            ),
            "validation_roc_auc": float(
                roc_auc_score(y_val, val_probabilities)
            ),
            "validation_pr_auc": float(
                average_precision_score(y_val, val_probabilities)
            ),
            "training_time_sec": float(training_seconds),
            "test_prediction_time_sec": float(prediction_seconds),
            **MODEL_PARAMETERS,
        })
        results.append(metrics)

        model_path = os.path.join(
            MODELS_DIR, f"phase16_{model_name}_model.joblib"
        )
        joblib.dump(model, model_path)

        if prediction_output is None:
            prediction_output = pd.DataFrame({
                "row_id": pd.read_csv(os.path.join(M3_DIR, "results", "test_audit.csv"), usecols=["row_id"])["row_id"].to_numpy(),
                "fake_label": y_test.to_numpy(),
            })

        prediction_output[f"{model_name}_probability"] = test_probabilities
        prediction_output[f"{model_name}_prediction"] = (
            test_probabilities >= selected_threshold
        ).astype(int)

        print(f"Feature count: {len(features)}")
        print(f"Training time: {training_seconds:.2f} seconds")
        print(f"Validation-selected threshold: {selected_threshold:.2f}")
        print(f"Test precision: {metrics['precision']:.6f}")
        print(f"Test recall:    {metrics['recall']:.6f}")
        print(f"Test F1:        {metrics['f1']:.6f}")
        print(f"Test ROC-AUC:   {metrics['roc_auc']:.6f}")
        print(f"Test PR-AUC:    {metrics['pr_auc']:.6f}")
        print(f"Confusion matrix: TN={metrics['tn']}, FP={metrics['fp']}, "
              f"FN={metrics['fn']}, TP={metrics['tp']}")
        print(f"Saved model: {model_path}")

    metrics_df = pd.DataFrame(results)
    thresholds_df = pd.concat(threshold_tables, ignore_index=True)

    # Include paired differences: semantic minus baseline.
    baseline = next(r for r in results if r["model"] == "baseline")
    semantic = next(r for r in results if r["model"] == "semantic")
    comparison = {}
    for metric in [
        "precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"
    ]:
        comparison[f"{metric}_difference_semantic_minus_baseline"] = (
            semantic[metric] - baseline[metric]
        )

    summary = {
        "phase": "16.4",
        "status": "completed",
        "experiment": "M3 baseline versus M4 semantic feature augmentation",
        "threshold_rule": (
            "Choose maximum validation F1 among thresholds "
            "[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]."
        ),
        "test_usage": "Test set used only after model training and validation threshold selection.",
        "missing_similarity_handling": (
            "Preserved as NaN; HistGradientBoostingClassifier native "
            "missing-value handling is used for the semantic model."
        ),
        "model_parameters": MODEL_PARAMETERS,
        "results": results,
        "semantic_minus_baseline": comparison,
        "caution": (
            "Compare models on the same test records. Differences are "
            "empirical results, not proof of generalization beyond this dataset."
        ),
    }

    metrics_df.to_csv(
        os.path.join(RESULTS_DIR, "phase16_model_comparison_metrics.csv"),
        index=False,
    )
    thresholds_df.to_csv(
        os.path.join(RESULTS_DIR, "phase16_model_comparison_validation_thresholds.csv"),
        index=False,
    )
    prediction_output.to_csv(
        os.path.join(RESULTS_DIR, "phase16_model_comparison_test_predictions.csv"),
        index=False,
    )
    with open(
        os.path.join(RESULTS_DIR, "phase16_model_comparison_summary.json"),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("\nSemantic minus baseline:")
    print(json.dumps(comparison, indent=2))
    print(f"\nResults saved under: {RESULTS_DIR}")

def main():
    audit_only = "--audit-only" in sys.argv

    print("=" * 70)
    print("PROJECT SENTINEL — PHASE 16 SEMANTIC MODEL COMPARISON")
    print("=" * 70)
    print("Mode:", "AUDIT ONLY" if audit_only else "TRAIN AND EVALUATE")

    data = load_datasets()

    # Convert names into a consistent mapping for each model.
    model_data = {}
    for split, pair in data.items():
        model_data[split] = {
            "baseline": pair["base"],
            "semantic": pair["semantic"],
        }

    audit = audit_datasets(data)
    audit_path = os.path.join(
        RESULTS_DIR, "phase16_model_comparison_input_audit.json"
    )
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    print("\nINPUT AUDIT: PASSED")
    print(f"Audit report: {audit_path}")

    if audit_only:
        print("Audit-only mode: no model was trained.")
        return

    run_experiment(model_data)

if __name__ == "__main__":
    main()
