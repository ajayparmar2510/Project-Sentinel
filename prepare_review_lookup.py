from pathlib import Path
import csv
import pandas as pd

ROOT = Path("/Users/apple/Downloads/YelpZip_Milestone2")
RAW = ROOT / "yelpzip_final.csv"
FEATURES = ROOT / "results/phase5_engineered_features.csv"
PREDICTIONS = Path("results/phase16_model_comparison_test_predictions.csv")
AUDIT = Path("../results/test_audit.csv")
OUTPUT = Path("results/review_lookup_test.csv")

print("Loading test predictions and audit records...")
pred = pd.read_csv(PREDICTIONS)
audit = pd.read_csv(AUDIT)
features = pd.read_csv(FEATURES)

lookup = (
    audit.merge(pred, on="row_id", validate="one_to_one")
    .merge(features, on="row_id", suffixes=("_audit", "_feature"), validate="one_to_one")
)

wanted = set(lookup["row_id"].astype(int))
print(f"Matched {len(wanted):,} test review IDs. Extracting original review text...")

with RAW.open("r", encoding="utf-8", newline="") as source:
    reader = csv.DictReader(source)
    rows = []
    for row_id, record in enumerate(reader, start=1):
        if row_id in wanted:
            rows.append({
                "row_id": row_id,
                "review_text": record.get("review_text", ""),
            })
        if row_id % 100000 == 0:
            print(f"Scanned {row_id:,} raw reviews...")
        if len(rows) == len(wanted):
            break

text = pd.DataFrame(rows)
lookup["row_id"] = lookup["row_id"].astype(int)
result = lookup.merge(text, on="row_id", how="left", validate="one_to_one")
result.to_csv(OUTPUT, index=False)

print(f"Saved: {OUTPUT}")
print(f"Rows: {len(result):,}")
print(f"Reviews with text: {result['review_text'].notna().sum():,}")
print("Columns include prediction scores, original text, and engineered evidence.")
