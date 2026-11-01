from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = Path(
    "/Users/apple/Downloads/YelpZip_Milestone2/yelpzip_final.csv"
)

FEATURE_FILE = Path(
    "/Users/apple/Downloads/YelpZip_Milestone2/results/phase5_engineered_features.csv"
)

DB_FILE = ROOT / "results" / "historical_reviews.sqlite3"

KEYS = ["user_id", "prod_id", "rating", "date"]

def normalize(frame):
    frame = frame.copy()

    for col in ["user_id", "prod_id"]:
        frame[col] = frame[col].astype("string").str.strip()

    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")

    frame["date"] = (
        pd.to_datetime(frame["date"], errors="coerce", format="mixed")
        .dt.strftime("%Y-%m-%d")
    )

    return frame

def main():
    if DB_FILE.exists():
        raise FileExistsError(
            f"Database already exists: {DB_FILE}\n"
            "It has not been overwritten. Rename or move it before rebuilding."
        )

    if not RAW_FILE.exists() or not FEATURE_FILE.exists():
        raise FileNotFoundError("One or both source CSV files are missing.")

    print("Loading original review data...")
    raw = pd.read_csv(
        RAW_FILE,
        usecols=["user_id", "prod_id", "rating", "date", "review_text"],
        dtype={"user_id": "string", "prod_id": "string"},
    )
    raw = normalize(raw)

    print("Loading historical feature data...")
    features = pd.read_csv(
        FEATURE_FILE,
        usecols=["row_id", "user_id", "prod_id", "rating", "date"],
        dtype={
            "row_id": "string",
            "user_id": "string",
            "prod_id": "string",
        },
    )
    features = normalize(features)
    features["row_id"] = features["row_id"].astype("string").str.strip()

    print("Checking keys and joining review text...")
    if raw[KEYS].isna().any().any():
        raise ValueError("Original dataset contains missing join-key values.")

    if features[KEYS].isna().any().any():
        raise ValueError("Feature dataset contains missing join-key values.")

    if raw.duplicated(KEYS).any():
        raise ValueError("Original dataset has duplicate composite keys.")

    if features.duplicated(KEYS).any():
        raise ValueError("Feature dataset has duplicate composite keys.")

    merged = features.merge(
        raw[KEYS + ["review_text"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(
            f"Join stopped: {unmatched} historical rows did not match."
        )

    if merged["review_text"].isna().any():
        raise ValueError("Some matched reviews have missing review text.")

    merged = merged[
        ["row_id", "user_id", "prod_id", "date", "rating", "review_text"]
    ].copy()

    merged.rename(columns={"prod_id": "product_id"}, inplace=True)

    print(f"Matched historical reviews: {len(merged):,}")
    print("Creating SQLite database...")

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE historical_reviews (
                row_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                date TEXT NOT NULL,
                rating REAL,
                review_text TEXT NOT NULL
            )
        """)

        merged.to_sql(
            "historical_reviews",
            conn,
            if_exists="append",
            index=False,
            chunksize=5000,
        )

        conn.execute("""
            CREATE INDEX idx_history_user_date
            ON historical_reviews(user_id, date)
        """)

        conn.execute("""
            CREATE INDEX idx_history_product
            ON historical_reviews(product_id)
        """)

        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM historical_reviews"
        ).fetchone()[0]

        sample = conn.execute("""
            SELECT row_id, user_id, date, rating, substr(review_text, 1, 100)
            FROM historical_reviews
            ORDER BY row_id
            LIMIT 3
        """).fetchall()

    if count != len(features):
        raise RuntimeError(
            f"Database verification failed: expected {len(features)}, got {count}"
        )

    print("\nDATABASE BUILD SUCCESSFUL")
    print(f"Rows stored: {count:,}")
    print(f"Database: {DB_FILE}")
    print("\nSample records:")
    for record in sample:
        print(record)

if __name__ == "__main__":
    main()
