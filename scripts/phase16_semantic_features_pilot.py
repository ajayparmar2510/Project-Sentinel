from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE = Path("/Users/apple/Downloads/YelpZip_Milestone2")
M4 = BASE / "Milestone3" / "Milestone4"
INPUT = BASE / "yelpzip_final.csv"
RESULTS = M4 / "results"
CACHE = M4 / "models" / "hf_cache"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_SIZE = 2000
SEED = 42
TRAIN_CUTOFF = pd.Timestamp("2014-01-07")

RESULTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

start = time.time()

print("PHASE 16.2: ROW-ALIGNED SEMANTIC FEATURE PILOT")
print("Input:", INPUT)
print("M4 output:", RESULTS)
print("MPS available:", torch.backends.mps.is_available())

# The original CSV's logical data-record position is the engineered row_id.
print("\nLoading source review records...")
df = pd.read_csv(
    INPUT,
    usecols=["user_id", "prod_id", "date", "review_text"],
    dtype={"user_id": str, "prod_id": str},
)
df.insert(0, "row_id", np.arange(1, len(df) + 1, dtype=np.int64))

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["date"] < TRAIN_CUTOFF].copy()
df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
df = df[df["review_text"].str.len() > 0].copy()

if len(df) < SAMPLE_SIZE:
    raise ValueError(f"Only {len(df)} usable training reviews found.")

sample = (
    df.sample(n=SAMPLE_SIZE, random_state=SEED)
    .sort_values(["date", "row_id"], kind="stable")
    .reset_index(drop=True)
)
del df

print("Sample size:", len(sample))
print("Sample date range:", sample["date"].min(), "to", sample["date"].max())

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(
    MODEL_NAME,
    device=device,
    cache_folder=str(CACHE),
)

print("Embedding reviews on:", device)
embeddings = model.encode(
    sample["review_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

if embeddings.shape != (SAMPLE_SIZE, 384):
    raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
if not np.isfinite(embeddings).all():
    raise ValueError("Non-finite embedding values detected.")

# For each review, compare only with earlier sampled reviews of the same product.
# Similarity uses text embeddings only; labels are not loaded or used.
max_prior_similarity = np.full(SAMPLE_SIZE, np.nan, dtype=np.float32)
prior_sampled_review_count = np.zeros(SAMPLE_SIZE, dtype=np.int32)

products = sample["prod_id"].fillna("").astype(str).to_numpy()
for i in range(SAMPLE_SIZE):
    prior_indices = np.flatnonzero(products[:i] == products[i])
    prior_sampled_review_count[i] = len(prior_indices)

    if len(prior_indices):
        similarities = embeddings[prior_indices] @ embeddings[i]
        max_prior_similarity[i] = float(np.max(similarities))

output = sample[["row_id", "user_id", "prod_id", "date"]].copy()
output["max_prior_sampled_product_similarity"] = max_prior_similarity
output["prior_sampled_product_review_count"] = prior_sampled_review_count

features_path = RESULTS / "phase16_semantic_features_pilot.csv"
embeddings_path = RESULTS / "phase16_semantic_features_pilot_embeddings.npy"
summary_path = RESULTS / "phase16_semantic_features_pilot_summary.json"

output.to_csv(features_path, index=False)
np.save(embeddings_path, embeddings)

summary = {
    "phase": "16.2",
    "status": "pilot",
    "model": MODEL_NAME,
    "device": device,
    "sample_size": int(len(sample)),
    "embedding_dimensions": int(embeddings.shape[1]),
    "embedding_values_finite": bool(np.isfinite(embeddings).all()),
    "row_id_unique": bool(output["row_id"].is_unique),
    "feature_rows": int(len(output)),
    "rows_with_prior_sampled_product_reviews": int(
        (prior_sampled_review_count > 0).sum()
    ),
    "similarity_values_defined": int(
        np.isfinite(max_prior_similarity).sum()
    ),
    "similarity_min": (
        float(np.nanmin(max_prior_similarity))
        if np.isfinite(max_prior_similarity).any()
        else None
    ),
    "similarity_max": (
        float(np.nanmax(max_prior_similarity))
        if np.isfinite(max_prior_similarity).any()
        else None
    ),
    "elapsed_seconds": round(time.time() - start, 2),
    "limitations": [
        "Similarity is computed only against earlier reviews in the sample.",
        "This is a pipeline pilot, not a full-dataset semantic feature set.",
        "No classifier performance claim is made.",
        "Rows without an earlier sampled review have undefined similarity."
    ],
}

summary_path.write_text(json.dumps(summary, indent=2))

print("\nPILOT VALIDATION")
print(json.dumps(summary, indent=2))
print("\nSaved:")
print(features_path)
print(embeddings_path)
print(summary_path)
