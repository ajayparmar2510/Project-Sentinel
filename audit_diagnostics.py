import os
import json
import joblib
import pandas as pd

ROOT = os.getcwd()
OUT = os.path.join(ROOT, "results", "sentinel_diagnostics")
os.makedirs(OUT, exist_ok=True)

files = [
    "results/phase16_model_comparison_test_predictions.csv",
    "results/review_lookup_test.csv",
    "results/phase18_reviewer_profiles_corrected.csv",
    "results/phase17_trust_intelligence_scores.csv",
    "results/reviewer_activity_summary.csv",
]

report = []

for file in files:
    path = os.path.join(ROOT, file)
    if not os.path.exists(path):
        report.append(f"\nMISSING: {file}")
        continue

    df = pd.read_csv(path, nrows=5, low_memory=False)
    report.append(f"\nFILE: {file}")
    report.append(f"COLUMNS: {list(df.columns)}")
    report.append(f"SAMPLE:\n{df.head(3).to_string(index=False)}")

    # Full row count without loading the complete file into memory
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        count = sum(1 for _ in f) - 1
    report.append(f"DATA ROWS: {count}")

for name in [
    "phase16_baseline_model.joblib",
    "phase16_semantic_model.joblib",
    "phase17_trust_logistic_model.joblib",
]:
    path = os.path.join(ROOT, "models", name)
    report.append(f"\nMODEL: {name}")
    try:
        model = joblib.load(path)
        report.append(f"TYPE: {type(model)}")
        report.append(f"CLASSES: {getattr(model, 'classes_', 'Not available')}")
        report.append(f"FEATURE COUNT: {getattr(model, 'n_features_in_', 'Not available')}")
        report.append(f"FEATURE NAMES: {getattr(model, 'feature_names_in_', 'Not available')}")
        if hasattr(model, "steps"):
            report.append(f"PIPELINE STEPS: {[step[0] for step in model.steps]}")
    except Exception as e:
        report.append(f"LOAD ERROR: {repr(e)}")

output = "\n".join(report)
report_path = os.path.join(OUT, "diagnostic_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(output)

print(output)
print(f"\nReport saved to: {report_path}")
