from pathlib import Path
from collections import defaultdict, deque
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
DIMENSIONS = 384
CHUNK_SIZE = 2000
WINDOW = 20

META_PATH = RESULTS / "phase16_full_metadata.csv"
EMBED_PATH = RESULTS / "phase16_full_embeddings.npy"
CHECKPOINT_PATH = RESULTS / "phase16_embedding_checkpoint.json"
FEATURE_PATH = RESULTS / "phase16_semantic_features_full.csv"
SUMMARY_PATH = RESULTS / "phase16_semantic_features_full_summary.json"

RESULTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

start = time.time()

print("=" * 65)
print("PHASE 16.3: FULL-DATASET SEMANTIC FEATURES")
print("=" * 65)
print("Source:", INPUT)
print("Model:", MODEL_NAME)
print("Window:", WINDOW)
print("MPS available:", torch.backends.mps.is_available())

# PASS 1: Preserve original CSV record IDs and build chronological metadata.
print("\n[1/3] Building chronological metadata...")

metadata_parts = []
offset = 0

for chunk in pd.read_csv(
    INPUT,
    usecols=["prod_id", "date"],
    dtype={"prod_id": "string", "date": "string"},
    chunksize=25000,
):
    row_ids = np.arange(offset + 1, offset + len(chunk) + 1,
                        dtype=np.int64)
    part = pd.DataFrame({
        "row_id": row_ids,
        "prod_id": chunk["prod_id"].fillna("").astype(str).to_numpy(),
        "date": pd.to_datetime(chunk["date"], errors="coerce").to_numpy(),
    })
    metadata_parts.append(part)
    offset += len(chunk)

metadata = pd.concat(metadata_parts, ignore_index=True)
del metadata_parts

if metadata["row_id"].duplicated().any():
    raise ValueError("Duplicate source row_id found.")
if metadata["date"].isna().any():
    raise ValueError("Invalid or missing dates found.")

metadata = metadata.sort_values(
    ["date", "row_id"], kind="stable"
).reset_index(drop=True)

metadata.to_csv(META_PATH, index=False)
total_rows = len(metadata)

print("Source rows:", f"{total_rows:,}")
print("Chronological range:",
      metadata["date"].min(), "to", metadata["date"].max())
print("Metadata saved:", META_PATH)

# PASS 2: Embed text in batches and write directly to a disk-backed array.
# The checkpoint permits resuming at a completed CSV chunk boundary.
print("\n[2/3] Generating embeddings in batches...")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(
    MODEL_NAME, device=device, cache_folder=str(CACHE)
)

completed_rows = 0
resume = False

if EMBED_PATH.exists() and CHECKPOINT_PATH.exists():
    try:
        checkpoint = json.loads(CHECKPOINT_PATH.read_text())
        existing = np.load(EMBED_PATH, mmap_mode="r")

        source_stat = INPUT.stat()
        completed = int(checkpoint.get("completed_rows", -1))
        valid_boundary = (
            completed == total_rows
            or (
                completed >= 0
                and completed % CHUNK_SIZE == 0
            )
        )

        if (
            checkpoint.get("model") == MODEL_NAME
            and checkpoint.get("total_rows") == total_rows
            and checkpoint.get("source_size") == source_stat.st_size
            and checkpoint.get("source_mtime_ns") == source_stat.st_mtime_ns
            and checkpoint.get("chunk_size") == CHUNK_SIZE
            and checkpoint.get("window_size") == WINDOW
            and existing.shape == (total_rows, DIMENSIONS)
            and valid_boundary
        ):
            completed_rows = completed
            resume = 0 <= completed_rows <= total_rows
    except Exception:
        resume = False

if resume:
    embeddings = np.load(EMBED_PATH, mmap_mode="r+")
    print("Resuming after", f"{completed_rows:,}", "rows.")
else:
    # Remove stale progress before starting a fresh embedding array.
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    embeddings = np.lib.format.open_memmap(
        EMBED_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, DIMENSIONS),
    )
    completed_rows = 0
    print("Starting a new embedding run.")

offset = 0
last_reported = completed_rows

for chunk in pd.read_csv(
    INPUT,
    usecols=["review_text"],
    dtype={"review_text": "string"},
    chunksize=CHUNK_SIZE,
):
    end = offset + len(chunk)

    if end <= completed_rows:
        offset = end
        continue

    if offset != completed_rows:
        raise ValueError(
            "Checkpoint is not at a valid chunk boundary. "
            "Remove the checkpoint and embedding file to restart."
        )

    texts = chunk["review_text"].fillna("").astype(str).str.strip().tolist()

    if any(not text for text in texts):
        raise ValueError(
            f"Empty review text encountered in rows "
            f"{offset + 1} through {end}."
        )

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if vectors.shape != (len(texts), DIMENSIONS):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"Non-finite embeddings in rows {offset + 1}-{end}.")

    embeddings[offset:end] = vectors.astype(np.float32, copy=False)
    embeddings.flush()

    completed_rows = end
    source_stat = INPUT.stat()
    checkpoint = {
        "model": MODEL_NAME,
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "embedding_dimensions": DIMENSIONS,
        "device": device,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "chunk_size": CHUNK_SIZE,
        "window_size": WINDOW,
    }
    temporary = CHECKPOINT_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(checkpoint, indent=2))
    temporary.replace(CHECKPOINT_PATH)

    offset = end

    if completed_rows - last_reported >= 10000 or completed_rows == total_rows:
        print(f"Embedded: {completed_rows:,}/{total_rows:,}")
        last_reported = completed_rows

del model
embeddings.flush()
print("Embedding array saved:", EMBED_PATH)

# PASS 3: Chronological, same-product comparisons.
# Only the immediately preceding WINDOW reviews are compared.
# No labels are loaded or used in this stage.
print("\n[3/3] Computing chronological product similarities...")

embeddings = np.load(EMBED_PATH, mmap_mode="r")
if embeddings.shape != (total_rows, DIMENSIONS):
    raise ValueError("Embedding array shape does not match source row count.")

history = defaultdict(lambda: deque(maxlen=WINDOW))
product_seen = defaultdict(int)

row_ids_out = np.empty(total_rows, dtype=np.int64)
similarities_out = np.full(total_rows, np.nan, dtype=np.float32)
prior_counts_out = np.zeros(total_rows, dtype=np.int32)

for position, record in enumerate(metadata.itertuples(index=False)):
    row_id = int(record.row_id)
    product = str(record.prod_id)

    # Missing product IDs are not grouped together.
    if not product:
        row_ids_out[position] = row_id
        continue

    prior_ids = history[product]
    prior_count = product_seen[product]
    row_ids_out[position] = row_id
    prior_counts_out[position] = prior_count

    if prior_count > 0:
        current = np.asarray(embeddings[row_id - 1], dtype=np.float32)
        prior_array = np.fromiter(prior_ids, dtype=np.int64) - 1
        prior_vectors = np.asarray(embeddings[prior_array], dtype=np.float32)
        similarities = prior_vectors @ current
        similarities_out[position] = np.max(similarities)

    prior_ids.append(row_id)
    product_seen[product] += 1

    if (position + 1) % 100000 == 0:
        print(f"Compared: {position + 1:,}/{total_rows:,}")

features = pd.DataFrame({
    "row_id": row_ids_out,
    "max_prior_20_product_similarity": similarities_out,
    "prior_product_review_count": prior_counts_out,
})

# Save in row_id order for easier, auditable joining.
features = features.sort_values("row_id", kind="stable").reset_index(drop=True)

if len(features) != total_rows:
    raise ValueError("Feature row count mismatch.")
if not features["row_id"].is_unique:
    raise ValueError("Feature row IDs are not unique.")
if not np.array_equal(
    features["row_id"].to_numpy(),
    np.arange(1, total_rows + 1, dtype=np.int64),
):
    raise ValueError("Feature row IDs do not cover the full source.")

defined = features["max_prior_20_product_similarity"].dropna()
if not defined.empty:
    if not np.isfinite(defined.to_numpy()).all():
        raise ValueError("Non-finite semantic similarity values found.")
    if (defined < -1.001).any() or (defined > 1.001).any():
        raise ValueError("Similarity outside cosine range.")

features.to_csv(FEATURE_PATH, index=False)

summary = {
    "phase": "16.3",
    "status": "completed",
    "model": MODEL_NAME,
    "device": device,
    "total_rows": int(total_rows),
    "embedding_dimensions": DIMENSIONS,
    "window_size": WINDOW,
    "row_id_unique": bool(features["row_id"].is_unique),
    "feature_rows": int(len(features)),
    "rows_with_prior_product_reviews": int(
        (features["prior_product_review_count"] > 0).sum()
    ),
    "rows_with_defined_similarity": int(features[
        "max_prior_20_product_similarity"
    ].notna().sum()),
    "rows_without_defined_similarity": int(features[
        "max_prior_20_product_similarity"
    ].isna().sum()),
    "similarity_min": float(defined.min()) if len(defined) else None,
    "similarity_max": float(defined.max()) if len(defined) else None,
    "elapsed_seconds": round(time.time() - start, 2),
    "limitations": [
        "Similarity compares only the previous 20 reviews for the same product.",
        "Reviews are ordered by date and then source row_id.",
        "Similarity is undefined when no eligible earlier product review exists.",
        "Semantic features do not use fraud labels.",
        "No classifier improvement is claimed by feature generation alone."
    ],
}

SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

print("\nVALIDATION SUMMARY")
print(json.dumps(summary, indent=2))
print("\nSaved feature file:", FEATURE_PATH)
print("Saved summary:", SUMMARY_PATH)
print("Embedding checkpoint:", CHECKPOINT_PATH)
print("Elapsed seconds:", round(time.time() - start, 2))
