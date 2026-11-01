from pathlib import Path
from collections import defaultdict, deque
import json
import time

import numpy as np
import pandas as pd

M4 = Path("/Users/apple/Downloads/YelpZip_Milestone2/Milestone3/Milestone4")
RESULTS = M4 / "results"

META_PATH = RESULTS / "phase16_full_metadata.csv"
EMBED_PATH = RESULTS / "phase16_full_embeddings.npy"
FEATURE_PATH = RESULTS / "phase16_semantic_features_strict_date.csv"
SUMMARY_PATH = RESULTS / "phase16_semantic_features_strict_date_summary.json"

WINDOW = 20
DIMENSIONS = 384
start = time.time()

print("=" * 65)
print("PHASE 16.3 CORRECTION: STRICTLY EARLIER DATES")
print("=" * 65)

metadata = pd.read_csv(
    META_PATH,
    dtype={"prod_id": "string"},
    parse_dates=["date"],
)
metadata["prod_id"] = metadata["prod_id"].fillna("").astype(str)

total_rows = len(metadata)
if metadata["row_id"].duplicated().any() or metadata["date"].isna().any():
    raise ValueError("Invalid metadata: duplicate row IDs or missing dates.")

embeddings = np.load(EMBED_PATH, mmap_mode="r")
if embeddings.shape != (total_rows, DIMENSIONS):
    raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

# Arrays are indexed by original source row_id minus one.
similarities_out = np.full(total_rows, np.nan, dtype=np.float32)
prior_counts_out = np.zeros(total_rows, dtype=np.int32)

history = defaultdict(lambda: deque(maxlen=WINDOW))
product_seen = defaultdict(int)

print("Rows:", f"{total_rows:,}")
print("Embedding shape:", embeddings.shape)
print("Window:", WINDOW)
print("Rule: only strictly earlier calendar dates")

# Process one date at a time. Every review on the current date is
# evaluated before any review from that date is added to history.
for date_number, (date, day) in enumerate(
    metadata.groupby("date", sort=True), start=1
):
    for product, group in day.groupby("prod_id", sort=False):
        if not product:
            continue

        row_ids = group["row_id"].to_numpy(dtype=np.int64)
        positions = row_ids - 1
        prior_ids = list(history[product])
        prior_count = product_seen[product]

        prior_counts_out[positions] = prior_count

        if prior_ids:
            current_vectors = np.asarray(
                embeddings[positions], dtype=np.float32
            )
            prior_positions = np.asarray(prior_ids, dtype=np.int64) - 1
            prior_vectors = np.asarray(
                embeddings[prior_positions], dtype=np.float32
            )

            scores = current_vectors @ prior_vectors.T
            scores = np.clip(scores, -1.0, 1.0)
            similarities_out[positions] = np.max(scores, axis=1)

        # Add this date's reviews only after all its comparisons.
        for row_id in row_ids:
            history[product].append(int(row_id))
        product_seen[product] += len(row_ids)

    if date_number % 250 == 0:
        print(
            f"Dates processed: {date_number:,}; "
            f"reviews covered: {int((metadata['date'] <= date).sum()):,}"
        )

features = pd.DataFrame({
    "row_id": np.arange(1, total_rows + 1, dtype=np.int64),
    "max_prior_20_product_similarity": similarities_out,
    "prior_product_review_count": prior_counts_out,
})

if len(features) != total_rows or not features["row_id"].is_unique:
    raise ValueError("Feature row coverage or uniqueness check failed.")

if not np.array_equal(
    features["row_id"].to_numpy(),
    np.arange(1, total_rows + 1, dtype=np.int64),
):
    raise ValueError("Feature row IDs do not cover the complete source.")

defined = features["max_prior_20_product_similarity"].dropna()
if not np.isfinite(defined.to_numpy()).all():
    raise ValueError("Non-finite similarity values found.")
if not defined.between(-1.0, 1.0).all():
    raise ValueError("Similarity outside [-1, 1].")

features.to_csv(FEATURE_PATH, index=False)

summary = {
    "phase": "16.3-corrected",
    "status": "completed",
    "total_rows": int(total_rows),
    "embedding_dimensions": DIMENSIONS,
    "window_size": WINDOW,
    "ordering_rule": (
        "Only strictly earlier calendar dates are eligible. "
        "Same-day reviews are excluded from comparisons and prior counts."
    ),
    "row_id_unique": bool(features["row_id"].is_unique),
    "feature_rows": int(len(features)),
    "rows_with_prior_product_reviews": int(
        (features["prior_product_review_count"] > 0).sum()
    ),
    "rows_with_defined_similarity": int(defined.notna().sum()),
    "rows_without_defined_similarity": int(defined.isna().sum()),
    "similarity_min": float(defined.min()) if len(defined) else None,
    "similarity_max": float(defined.max()) if len(defined) else None,
    "similarity_values_clipped_to_range": True,
    "elapsed_seconds": round(time.time() - start, 2),
    "source_embeddings_reused": str(EMBED_PATH),
    "feature_file": str(FEATURE_PATH),
    "note": (
        "Feature generation only; classifier improvement has not yet "
        "been evaluated."
    ),
}

SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

print("\nVALIDATION SUMMARY")
print(json.dumps(summary, indent=2))
print("\nSaved corrected features:", FEATURE_PATH)
print("Saved summary:", SUMMARY_PATH)
