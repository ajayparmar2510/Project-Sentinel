import pandas as pd

root = "results/"

reviews = pd.read_csv(root + "review_lookup_test.csv", low_memory=False)
profiles = pd.read_csv(root + "phase18_reviewer_profiles_corrected.csv")
activity = pd.read_csv(root + "reviewer_activity_summary.csv")

print("\n========== REVIEW ID CHECK ==========")

for rid in [19885, 1199]:
    row = reviews[reviews["row_id"] == rid]

    if row.empty:
        print(f"\nReview ID {rid}: NOT FOUND")
        continue

    cols = [
        "row_id", "user_id_audit", "user_id_feature",
        "prod_id_audit", "prod_id_feature",
        "date_audit", "date_feature",
        "fake_label_audit", "fake_label_feature",
        "semantic_probability", "semantic_prediction",
        "user_prior_reviews", "user_prior_reviews_30d",
        "days_since_user_previous_review"
    ]

    print(f"\nReview ID {rid}:")
    print(row[cols].to_string(index=False))

    uid = row.iloc[0]["user_id_audit"]

    print("\nMatching reviewer activity:")
    print(activity[activity["user_id"] == uid].to_string(index=False))

    print("\nMatching reviewer profile:")
    print(profiles[profiles["user_id"] == uid].to_string(index=False))

print("\n========== REVIEWER ID CHECK ==========")

for uid in [8367, 8306]:
    print(f"\nReviewer ID {uid}:")

    print("\nActivity:")
    print(activity[activity["user_id"] == uid].to_string(index=False))

    print("\nProfile:")
    print(profiles[profiles["user_id"] == uid].to_string(index=False))

    reviewer_reviews = reviews[reviews["user_id_audit"] == uid]

    print("\nSaved test predictions for this reviewer:", len(reviewer_reviews))

    if not reviewer_reviews.empty:
        print(reviewer_reviews[
            ["row_id", "date_audit", "semantic_probability",
             "user_prior_reviews", "user_prior_reviews_30d",
             "days_since_user_previous_review"]
        ].head(10).to_string(index=False))

print("\n========== ID AND LABEL CONSISTENCY ==========")
print("Review ID matches:",
      (reviews["user_id_audit"] == reviews["user_id_feature"]).all(),
      (reviews["prod_id_audit"] == reviews["prod_id_feature"]).all(),
      (reviews["date_audit"].astype(str) == reviews["date_feature"].astype(str)).all(),
      (reviews["fake_label_audit"] == reviews["fake_label_feature"]).all())

print("\nVerification completed.")
