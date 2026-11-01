from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

PROFILE_FILE = RESULTS / "phase18_reviewer_profiles_corrected.csv"
RISK_FILE = RESULTS / "phase17_trust_intelligence_scores.csv"
LOOKUP_FILE = RESULTS / "review_lookup_test.csv"
HISTORICAL_FILE = Path(
    "/Users/apple/Downloads/YelpZip_Milestone2/results/"
    "phase5_engineered_features.csv"
)

app = FastAPI(
    title="Project Sentinel API",
    description="Saved review predictions, reviewer profiles, and historical evidence.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def clean(value):
    """Convert pandas values into JSON-safe Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def clean_record(record):
    return {key: clean(value) for key, value in record.items()}


profiles = pd.read_csv(PROFILE_FILE, dtype={"user_id": str})
risk_scores = pd.read_csv(RISK_FILE, dtype={"user_id": str})

lookup = pd.read_csv(
    LOOKUP_FILE,
    dtype={
        "row_id": str,
        "user_id_feature": str,
        "prod_id_feature": str,
    },
)

historical_columns = [
    "row_id",
    "user_id",
    "prod_id",
    "date",
    "rating",
    "fake_label",
    "user_prior_reviews",
    "user_prior_reviews_30d",
    "days_since_user_previous_review",
    "review_text_is_duplicate",
    "review_text_reuse_ratio",
    "user_rating_deviation",
    "user_product_diversity",
    "repeated_word_ratio",
    "product_prior_reviews",
    "product_rating_deviation",
]

historical = pd.read_csv(
    HISTORICAL_FILE,
    usecols=lambda column: column in historical_columns,
    dtype={"row_id": str, "user_id": str, "prod_id": str},
)

for frame in (profiles, risk_scores):
    if "user_id" in frame.columns:
        frame["user_id"] = frame["user_id"].astype(str).str.strip()

for column in ("user_id_feature", "prod_id_feature", "row_id"):
    if column in lookup.columns:
        lookup[column] = lookup[column].astype(str).str.strip()

if "row_id" in historical.columns:
    historical["row_id"] = historical["row_id"].astype(str).str.strip()

if "user_id" in historical.columns:
    historical["user_id"] = historical["user_id"].astype(str).str.strip()

if "prod_id" in historical.columns:
    historical["prod_id"] = historical["prod_id"].astype(str).str.strip()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "Project Sentinel API",
        "reviewer_profiles_loaded": len(profiles),
        "review_risk_records_loaded": len(risk_scores),
        "saved_test_predictions_loaded": len(lookup),
        "historical_reviews_loaded": len(historical),
    }


@app.get("/api/reviewers/{user_id}")
def get_reviewer(user_id: str):
    user_id = user_id.strip()
    matched = profiles[profiles["user_id"] == user_id]

    if not user_id or matched.empty:
        raise HTTPException(
            status_code=404,
            detail="Reviewer ID was not found in the saved reviewer profiles.",
        )

    reviewer = matched.iloc[0]
    reviewer_risks = risk_scores[risk_scores["user_id"] == user_id]
    reviewer_risks = reviewer_risks.dropna(
        subset=["trust_intelligence_risk_score"]
    )
    recent_risks = reviewer_risks.sort_values(
        "date", ascending=False
    ).head(10)

    def number(value):
        return None if pd.isna(value) else float(value)

    return {
        "user_id": user_id,
        "review_count": int(reviewer["review_count"]),
        "mean_risk_index": number(reviewer["mean_risk_index"]),
        "median_risk_index": number(reviewer["median_risk_index"]),
        "max_risk_index": number(reviewer["max_risk_index"]),
        "risk_index_std": number(reviewer["risk_index_std"]),
        "high_risk_review_count": int(reviewer["high_risk_review_count"]),
        "distinct_other_entities": int(reviewer["distinct_other_entities"]),
        "first_review_date": clean(reviewer["first_review_date"]),
        "last_review_date": clean(reviewer["last_review_date"]),
        "high_risk_review_pct": number(reviewer["high_risk_review_pct"]),
        "risk_records": [
            {
                "date": clean(row["date"]),
                "product_id": clean(row["prod_id"]),
                "risk_score": number(row["trust_intelligence_risk_score"]),
                "interpretation": clean(row["score_interpretation"]),
            }
            for _, row in recent_risks.iterrows()
        ],
        "notice": (
            "Risk indicators are analytical signals, not proof of fraud "
            "or misconduct."
        ),
    }


@app.get("/api/reviews")
def get_reviews(
    limit: int = Query(default=50, ge=1, le=200),
    reviewer_id: str | None = None,
):
    """Return saved held-out test predictions, ordered by model score."""
    cases = lookup.copy()

    if reviewer_id:
        cases = cases[
            cases["user_id_feature"] == reviewer_id.strip()
        ]

    cases = cases.dropna(subset=["semantic_probability"])
    cases = cases.sort_values(
        "semantic_probability", ascending=False
    ).head(limit)

    output = []
    for _, row in cases.iterrows():
        output.append({
            "row_id": clean(row.get("row_id")),
            "user_id": clean(row.get("user_id_feature")),
            "product_id": clean(row.get("prod_id_feature")),
            "date": clean(row.get("date_feature")),
            "rating": clean(row.get("rating")),
            "semantic_probability": clean(row.get("semantic_probability")),
            "semantic_prediction": clean(row.get("semantic_prediction")),
            "review_text": clean(row.get("review_text")),
        })

    return {
        "count": len(output),
        "items": output,
        "notice": (
            "These are saved held-out test predictions, not predictions "
            "for every historical review."
        ),
    }


@app.get("/api/reviews/{row_id}")
def get_review(row_id: str):
    """Return a saved test prediction and its historical evidence."""
    matches = lookup[lookup["row_id"] == row_id.strip()]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No saved test prediction exists for this review ID."
            ),
        )

    row = matches.iloc[0]
    historical_matches = historical[
        historical["row_id"] == row_id.strip()
    ]

    evidence = (
        clean_record(historical_matches.iloc[0].to_dict())
        if not historical_matches.empty
        else {}
    )

    return {
        "row_id": clean(row.get("row_id")),
        "user_id": clean(row.get("user_id_feature")),
        "product_id": clean(row.get("prod_id_feature")),
        "date": clean(row.get("date_feature")),
        "rating": clean(row.get("rating")),
        "review_text": clean(row.get("review_text")),
        "semantic_probability": clean(row.get("semantic_probability")),
        "semantic_prediction": clean(row.get("semantic_prediction")),
        "baseline_probability": clean(row.get("baseline_probability")),
        "baseline_prediction": clean(row.get("baseline_prediction")),
        "saved_reference_label": clean(row.get("fake_label_feature")),
        "saved_test_features": clean_record(row.to_dict()),
        "historical_evidence": evidence,
        "notice": (
            "The model score is an analytical signal, not proof of fraud. "
            "The reference label is a dataset label, not an independent "
            "finding of misconduct."
        ),
    }


@app.get("/api/reviewers/{user_id}/reviews")
def get_reviewer_reviews(user_id: str):
    """Return saved test predictions associated with a reviewer."""
    user_id = user_id.strip()

    cases = lookup[
        lookup["user_id_feature"] == user_id
    ].copy()

    if cases.empty:
        return {
            "user_id": user_id,
            "count": 0,
            "items": [],
            "notice": "No saved test predictions were found for this reviewer.",
        }

    cases = cases.dropna(subset=["semantic_probability"])
    cases = cases.sort_values(
        "semantic_probability", ascending=False
    )

    return {
        "user_id": user_id,
        "count": len(cases),
        "items": [
            {
                "row_id": clean(row.get("row_id")),
                "product_id": clean(row.get("prod_id_feature")),
                "date": clean(row.get("date_feature")),
                "rating": clean(row.get("rating")),
                "semantic_probability": clean(row.get("semantic_probability")),
                "semantic_prediction": clean(row.get("semantic_prediction")),
                "review_text": clean(row.get("review_text")),
            }
            for _, row in cases.iterrows()
        ],
        "notice": (
            "Only saved held-out test predictions are listed here."
        ),
    }
