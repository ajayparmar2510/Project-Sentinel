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

RESULTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_SIZE = 2000
SEED = 42

print("PHASE 16.1: SEMANTIC MODEL PILOT")
print("Input:", INPUT)
print("Model:", MODEL_NAME)
print("MPS available:", torch.backends.mps.is_available())

start = time.time()

print("\nLoading training-period review text...")
df = pd.read_csv(
    INPUT,
    usecols=["user_id", "prod_id", "date", "review_text"],
    dtype={"user_id": str, "prod_id": str},
)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["date"] < pd.Timestamp("2014-01-07")].copy()
df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
df = df[df["review_text"].str.len() > 0]

if len(df) < SAMPLE_SIZE:
    raise ValueError(f"Only {len(df)} usable training reviews found.")

sample = df.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
del df

print("Pilot reviews:", len(sample))
print("Loading sentence embedding model...")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(
    MODEL_NAME,
    device=device,
    cache_folder=str(CACHE),
)

print("Device:", device)
print("Generating embeddings...")

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
    raise ValueError("Embedding matrix contains non-finite values.")

embedding_path = RESULTS / "phase16_pilot_embeddings.npy"
metadata_path = RESULTS / "phase16_pilot_metadata.csv"
summary_path = RESULTS / "phase16_pilot_summary.json"

np.save(embedding_path, embeddings)
sample.drop(columns=["review_text"]).to_csv(metadata_path, index=False)

summary = {
    "phase": "16.1",
    "model": MODEL_NAME,
    "device": device,
    "sample_size": int(len(sample)),
    "embedding_dimensions": int(embeddings.shape[1]),
    "embeddings_finite": bool(np.isfinite(embeddings).all()),
    "embedding_norm_mean": float(
        np.linalg.norm(embeddings, axis=1).mean()
    ),
    "elapsed_seconds": round(time.time() - start, 2),
    "note": (
        "Pilot only. No classifier evaluation or fraud-performance "
        "claim is made from this sample."
    ),
}

summary_path.write_text(json.dumps(summary, indent=2))

print("\nPILOT VALIDATION: PASSED")
print(json.dumps(summary, indent=2))
print("\nSaved:")
print(embedding_path)
print(metadata_path)
print(summary_path)
