from pathlib import Path
import pandas as pd

source = Path(
    "/Users/apple/Downloads/YelpZip_Milestone2/results/"
    "phase5_engineered_features.csv"
)
output = Path("results/reviewer_activity_summary.csv")

usecols = [
    "user_id",
    "rating",
    "date",
    "user_burstiness_flag",
    "user_prior_reviews",
    "user_prior_reviews_30d",
    "review_text_is_duplicate",
    "semantic_probability",
    "semantic_prediction",
]

header = pd.read_csv(source, nrows=0).columns.tolist()
available = [col for col in usecols if col in header]
required = ["user_id", "rating", "date"]
missing = [col for col in required if col not in available]

if missing:
    raise SystemExit(f"Missing required columns: {missing}")

df = pd.read_csv(source, usecols=available, low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

agg = {
    "review_count": ("user_id", "size"),
    "average_rating": ("rating", "mean"),
    "first_review": ("date", "min"),
    "last_review": ("date", "max"),
}

if "user_burstiness_flag" in df:
    agg["burst_flag_count"] = ("user_burstiness_flag", "sum")
if "review_text_is_duplicate" in df:
    agg["duplicate_review_count"] = ("review_text_is_duplicate", "sum")
if "semantic_prediction" in df:
    agg["model_flagged_count"] = ("semantic_prediction", "sum")
if "semantic_probability" in df:
    agg["mean_estimated_risk"] = ("semantic_probability", "mean")

summary = df.groupby("user_id").agg(**agg).reset_index()
summary["review_span_days"] = (
    summary["last_review"] - summary["first_review"]
).dt.days
summary["review_count_band"] = pd.cut(
    summary["review_count"],
    bins=[0, 10, 25, 50, 100, float("inf")],
    labels=["1–10", "11–25", "26–50", "51–100", "101+"],
)
summary = summary.sort_values("review_count", ascending=False)
output.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(output, index=False)

print("Reviewer summary created:", output)
print("Unique reviewers:", len(summary))
print("Reviewers with more than 50 reviews:", int((summary["review_count"] > 50).sum()))
print("\nTop 10 reviewers:")
print(summary.head(10).to_string(index=False))
print("\nAvailable summary columns:", ", ".join(summary.columns))
