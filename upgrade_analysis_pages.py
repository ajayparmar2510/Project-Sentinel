from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

start = source.find('elif section == "Reviewer Analysis":')
end = source.find('elif section == "Model Comparison":', start)

if start == -1 or end == -1:
    raise SystemExit("Could not find the analysis-page boundaries. No changes made.")

replacement = r'''elif section == "Reviewer Analysis":
    st.header("Reviewer Intelligence")
    st.caption(
        "Investigate full-dataset reviewer activity, then inspect saved model "
        "predictions for that reviewer's held-out test examples."
    )

    summary_path = RESULTS / "reviewer_activity_summary.csv"
    full_path = DATASET_PATH
    lookup_path = RESULTS / "review_lookup_test.csv"

    if not summary_path.exists() or not full_path.exists():
        st.error("Reviewer summary or full engineered dataset is missing.")
    else:
        @st.cache_data(show_spinner="Loading reviewer intelligence...")
        def load_reviewer_summary(path):
            return pd.read_csv(path, low_memory=False)

        @st.cache_data(show_spinner="Loading reviewer history...")
        def load_full_reviewer_data(path):
            cols = [
                "user_id", "rating", "date", "user_burstiness_flag",
                "review_text_is_duplicate", "user_prior_reviews_30d",
            ]
            available = pd.read_csv(path, nrows=0).columns.tolist()
            return pd.read_csv(
                path, usecols=[c for c in cols if c in available],
                low_memory=False
            )

        summary = load_reviewer_summary(str(summary_path))
        full = load_full_reviewer_data(str(full_path))
        summary["user_id"] = summary["user_id"].astype(str)
        full["user_id"] = full["user_id"].astype(str)
        full["date"] = pd.to_datetime(full["date"], errors="coerce")

        st.markdown("### Find a reviewer")
        min_reviews = st.select_slider(
            "Minimum number of reviews",
            options=[1, 10, 25, 50, 75, 100],
            value=50,
        )
        eligible = summary[summary["review_count"] >= min_reviews].copy()
        eligible = eligible.sort_values("review_count", ascending=False)

        st.caption(
            f"{len(eligible):,} reviewers meet the selected minimum. "
            "Review volume is an investigation filter, not proof of deception."
        )
        if eligible.empty:
            st.warning("No reviewers meet this activity threshold.")
        else:
            eligible["selector_label"] = eligible.apply(
                lambda r: f"User {r['user_id']} · {int(r['review_count'])} reviews",
                axis=1,
            )
            selected_label = st.selectbox(
                "Select reviewer",
                eligible["selector_label"].tolist(),
            )
            selected_user = selected_label.split(" · ")[0].replace("User ", "")
            profile = summary[summary["user_id"] == selected_user].iloc[0]
            history = full[full["user_id"] == selected_user].copy()
            history = history.sort_values("date")

            st.markdown("### Reviewer profile")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total reviews", f"{int(profile['review_count']):,}")
            c2.metric("Average rating", f"{profile['average_rating']:.2f}/5")
            c3.metric("Burst-flagged reviews", f"{int(profile.get('burst_flag_count', 0)):,}")
            c4.metric("Duplicate-text reviews", f"{int(profile.get('duplicate_review_count', 0)):,}")

            st.caption(
                f"Observed history: {profile['first_review']} to "
                f"{profile['last_review']} · "
                f"{int(profile['review_span_days']) if pd.notna(profile['review_span_days']) else 0:,} days"
            )

            st.markdown("### Review activity timeline")
            if not history.empty and history["date"].notna().any():
                monthly = (
                    history.dropna(subset=["date"])
                    .set_index("date").resample("MS").size()
                    .rename("Reviews")
                )
                st.area_chart(monthly)
                st.caption(
                    "A concentrated period may merit closer inspection. "
                    "Activity alone does not establish coordinated or fake reviews."
                )
            else:
                st.info("No valid dates were available for this reviewer.")

            st.markdown("### Rating distribution")
            if "rating" in history.columns:
                ratings = history["rating"].value_counts().sort_index()
                st.bar_chart(ratings.rename_axis("Rating").rename("Reviews"))

            st.markdown("### Individual review evidence")
            if lookup_path.exists():
                @st.cache_data(show_spinner="Loading held-out predictions...")
                def load_prediction_lookup(path):
                    return pd.read_csv(path, low_memory=False)

                lookup = load_prediction_lookup(str(lookup_path))
                lookup["user_id_feature"] = lookup["user_id_feature"].astype(str)
                individual = lookup[
                    lookup["user_id_feature"] == selected_user
                ].copy()

                if individual.empty:
                    st.info(
                        "This reviewer has no rows in the held-out test lookup. "
                        "Their full-dataset activity is shown above, but no "
                        "test-set model prediction is available to display."
                    )
                else:
                    flagged = int((individual["semantic_prediction"] == 1).sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Test-set reviews", f"{len(individual):,}")
                    c2.metric("Model-flagged test reviews", f"{flagged:,}")
                    c3.metric(
                        "Mean estimated risk",
                        f"{individual['semantic_probability'].mean():.1%}",
                    )

                    show_cols = [
                        c for c in [
                            "row_id", "date_feature", "rating",
                            "semantic_probability", "semantic_prediction",
                            "user_burstiness_flag", "review_text_is_duplicate",
                            "review_text_reuse_ratio", "user_rating_deviation",
                            "product_rating_deviation", "review_text",
                        ] if c in individual.columns
                    ]
                    st.dataframe(
                        individual[show_cols].sort_values(
                            "semantic_probability", ascending=False
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    st.markdown("### How to interpret a flagged review")
                    st.write(
                        "The model score prioritizes reviews for human inspection. "
                        "Burst flags, duplicate-text indicators, and rating deviations "
                        "are contextual signals, not proof of deception. The table "
                        "shows saved held-out predictions rather than live inference."
                    )
            else:
                st.warning("Held-out prediction lookup file is missing.")

elif section == "Temporal Analysis":
    st.header("Temporal Intelligence")
    st.caption(
        "Explore review activity over the full dataset. Model-risk trends are "
        "shown separately using saved held-out test predictions."
    )

    if not DATASET_PATH.exists():
        st.error("The full engineered dataset was not found.")
    else:
        @st.cache_data(show_spinner="Loading temporal review data...")
        def load_temporal_full(path):
            cols = ["user_id", "date", "rating"]
            available = pd.read_csv(path, nrows=0).columns.tolist()
            return pd.read_csv(
                path, usecols=[c for c in cols if c in available],
                low_memory=False
            )

        temporal = load_temporal_full(str(DATASET_PATH))
        temporal["date"] = pd.to_datetime(temporal["date"], errors="coerce")
        temporal = temporal.dropna(subset=["date"]).copy()

        if temporal.empty:
            st.warning("No valid review dates were found.")
        else:
            granularity = st.radio(
                "Time grouping",
                ["Monthly", "Quarterly", "Yearly"],
                horizontal=True,
            )
            if granularity == "Monthly":
                temporal["period"] = temporal["date"].dt.to_period("M").astype(str)
            elif granularity == "Quarterly":
                temporal["period"] = temporal["date"].dt.to_period("Q").astype(str)
            else:
                temporal["period"] = temporal["date"].dt.to_period("Y").astype(str)

            trend = temporal.groupby("period").agg(
                reviews=("user_id", "size"),
                unique_reviewers=("user_id", "nunique"),
                average_rating=("rating", "mean"),
            ).sort_index()

            c1, c2, c3 = st.columns(3)
            c1.metric("Reviews with valid dates", f"{len(temporal):,}")
            c2.metric("Time periods", f"{len(trend):,}")
            c3.metric("Unique reviewers", f"{temporal['user_id'].nunique():,}")

            st.markdown("### Review volume over time")
            st.area_chart(trend["reviews"])
            st.markdown("### Active reviewers over time")
            st.line_chart(trend["unique_reviewers"])
            st.markdown("### Average rating over time")
            st.line_chart(trend["average_rating"])

            st.markdown("### Saved model predictions over time")
            lookup_path = RESULTS / "review_lookup_test.csv"
            if lookup_path.exists():
                @st.cache_data(show_spinner="Loading held-out risk trends...")
                def load_temporal_predictions(path):
                    return pd.read_csv(
                        path,
                        usecols=[
                            "date_feature", "semantic_probability",
                            "semantic_prediction",
                        ],
                        low_memory=False,
                    )

                predictions = load_temporal_predictions(str(lookup_path))
                predictions["date"] = pd.to_datetime(
                    predictions["date_feature"], errors="coerce"
                )
                predictions = predictions.dropna(subset=["date"]).copy()
                if granularity == "Monthly":
                    predictions["period"] = predictions["date"].dt.to_period("M").astype(str)
                elif granularity == "Quarterly":
                    predictions["period"] = predictions["date"].dt.to_period("Q").astype(str)
                else:
                    predictions["period"] = predictions["date"].dt.to_period("Y").astype(str)

                risk_trend = predictions.groupby("period").agg(
                    mean_estimated_risk=("semantic_probability", "mean"),
                    model_flagged=("semantic_prediction", "sum"),
                    test_reviews=("semantic_prediction", "size"),
                ).sort_index()

                st.line_chart(risk_trend["mean_estimated_risk"])
                st.bar_chart(risk_trend["model_flagged"])
                with st.expander("View temporal data tables"):
                    st.dataframe(trend.reset_index(), width="stretch", hide_index=True)
                    st.dataframe(risk_trend.reset_index(), width="stretch", hide_index=True)
                st.caption(
                    "Risk and flagged-review trends use the held-out test subset, "
                    "not predictions for every review in the full dataset."
                )
            else:
                st.dataframe(trend.reset_index(), width="stretch", hide_index=True)
                st.info("Test prediction lookup is unavailable; risk trends are omitted.")
'''

path.write_text(source[:start] + replacement + source[end:], encoding="utf-8")
print("Updated Reviewer Analysis and Temporal Analysis.")
PYcat > upgrade_analysis_pages.py <<'PY'
from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

start = source.find('elif section == "Reviewer Analysis":')
end = source.find('elif section == "Model Comparison":', start)

if start == -1 or end == -1:
    raise SystemExit("Could not find the analysis-page boundaries. No changes made.")

replacement = r'''elif section == "Reviewer Analysis":
    st.header("Reviewer Intelligence")
    st.caption(
        "Investigate full-dataset reviewer activity, then inspect saved model "
        "predictions for that reviewer's held-out test examples."
    )

    summary_path = RESULTS / "reviewer_activity_summary.csv"
    full_path = DATASET_PATH
    lookup_path = RESULTS / "review_lookup_test.csv"

    if not summary_path.exists() or not full_path.exists():
        st.error("Reviewer summary or full engineered dataset is missing.")
    else:
        @st.cache_data(show_spinner="Loading reviewer intelligence...")
        def load_reviewer_summary(path):
            return pd.read_csv(path, low_memory=False)

        @st.cache_data(show_spinner="Loading reviewer history...")
        def load_full_reviewer_data(path):
            cols = [
                "user_id", "rating", "date", "user_burstiness_flag",
                "review_text_is_duplicate", "user_prior_reviews_30d",
            ]
            available = pd.read_csv(path, nrows=0).columns.tolist()
            return pd.read_csv(
                path, usecols=[c for c in cols if c in available],
                low_memory=False
            )

        summary = load_reviewer_summary(str(summary_path))
        full = load_full_reviewer_data(str(full_path))
        summary["user_id"] = summary["user_id"].astype(str)
        full["user_id"] = full["user_id"].astype(str)
        full["date"] = pd.to_datetime(full["date"], errors="coerce")

        st.markdown("### Find a reviewer")
        min_reviews = st.select_slider(
            "Minimum number of reviews",
            options=[1, 10, 25, 50, 75, 100],
            value=50,
        )
        eligible = summary[summary["review_count"] >= min_reviews].copy()
        eligible = eligible.sort_values("review_count", ascending=False)

        st.caption(
            f"{len(eligible):,} reviewers meet the selected minimum. "
            "Review volume is an investigation filter, not proof of deception."
        )
        if eligible.empty:
            st.warning("No reviewers meet this activity threshold.")
        else:
            eligible["selector_label"] = eligible.apply(
                lambda r: f"User {r['user_id']} · {int(r['review_count'])} reviews",
                axis=1,
            )
            selected_label = st.selectbox(
                "Select reviewer",
                eligible["selector_label"].tolist(),
            )
            selected_user = selected_label.split(" · ")[0].replace("User ", "")
            profile = summary[summary["user_id"] == selected_user].iloc[0]
            history = full[full["user_id"] == selected_user].copy()
            history = history.sort_values("date")

            st.markdown("### Reviewer profile")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total reviews", f"{int(profile['review_count']):,}")
            c2.metric("Average rating", f"{profile['average_rating']:.2f}/5")
            c3.metric("Burst-flagged reviews", f"{int(profile.get('burst_flag_count', 0)):,}")
            c4.metric("Duplicate-text reviews", f"{int(profile.get('duplicate_review_count', 0)):,}")

            st.caption(
                f"Observed history: {profile['first_review']} to "
                f"{profile['last_review']} · "
                f"{int(profile['review_span_days']) if pd.notna(profile['review_span_days']) else 0:,} days"
            )

            st.markdown("### Review activity timeline")
            if not history.empty and history["date"].notna().any():
                monthly = (
                    history.dropna(subset=["date"])
                    .set_index("date").resample("MS").size()
                    .rename("Reviews")
                )
                st.area_chart(monthly)
                st.caption(
                    "A concentrated period may merit closer inspection. "
                    "Activity alone does not establish coordinated or fake reviews."
                )
            else:
                st.info("No valid dates were available for this reviewer.")

            st.markdown("### Rating distribution")
            if "rating" in history.columns:
                ratings = history["rating"].value_counts().sort_index()
                st.bar_chart(ratings.rename_axis("Rating").rename("Reviews"))

            st.markdown("### Individual review evidence")
            if lookup_path.exists():
                @st.cache_data(show_spinner="Loading held-out predictions...")
                def load_prediction_lookup(path):
                    return pd.read_csv(path, low_memory=False)

                lookup = load_prediction_lookup(str(lookup_path))
                lookup["user_id_feature"] = lookup["user_id_feature"].astype(str)
                individual = lookup[
                    lookup["user_id_feature"] == selected_user
                ].copy()

                if individual.empty:
                    st.info(
                        "This reviewer has no rows in the held-out test lookup. "
                        "Their full-dataset activity is shown above, but no "
                        "test-set model prediction is available to display."
                    )
                else:
                    flagged = int((individual["semantic_prediction"] == 1).sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Test-set reviews", f"{len(individual):,}")
                    c2.metric("Model-flagged test reviews", f"{flagged:,}")
                    c3.metric(
                        "Mean estimated risk",
                        f"{individual['semantic_probability'].mean():.1%}",
                    )

                    show_cols = [
                        c for c in [
                            "row_id", "date_feature", "rating",
                            "semantic_probability", "semantic_prediction",
                            "user_burstiness_flag", "review_text_is_duplicate",
                            "review_text_reuse_ratio", "user_rating_deviation",
                            "product_rating_deviation", "review_text",
                        ] if c in individual.columns
                    ]
                    st.dataframe(
                        individual[show_cols].sort_values(
                            "semantic_probability", ascending=False
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    st.markdown("### How to interpret a flagged review")
                    st.write(
                        "The model score prioritizes reviews for human inspection. "
                        "Burst flags, duplicate-text indicators, and rating deviations "
                        "are contextual signals, not proof of deception. The table "
                        "shows saved held-out predictions rather than live inference."
                    )
            else:
                st.warning("Held-out prediction lookup file is missing.")

elif section == "Temporal Analysis":
    st.header("Temporal Intelligence")
    st.caption(
        "Explore review activity over the full dataset. Model-risk trends are "
        "shown separately using saved held-out test predictions."
    )

    if not DATASET_PATH.exists():
        st.error("The full engineered dataset was not found.")
    else:
        @st.cache_data(show_spinner="Loading temporal review data...")
        def load_temporal_full(path):
            cols = ["user_id", "date", "rating"]
            available = pd.read_csv(path, nrows=0).columns.tolist()
            return pd.read_csv(
                path, usecols=[c for c in cols if c in available],
                low_memory=False
            )

        temporal = load_temporal_full(str(DATASET_PATH))
        temporal["date"] = pd.to_datetime(temporal["date"], errors="coerce")
        temporal = temporal.dropna(subset=["date"]).copy()

        if temporal.empty:
            st.warning("No valid review dates were found.")
        else:
            granularity = st.radio(
                "Time grouping",
                ["Monthly", "Quarterly", "Yearly"],
                horizontal=True,
            )
            if granularity == "Monthly":
                temporal["period"] = temporal["date"].dt.to_period("M").astype(str)
            elif granularity == "Quarterly":
                temporal["period"] = temporal["date"].dt.to_period("Q").astype(str)
            else:
                temporal["period"] = temporal["date"].dt.to_period("Y").astype(str)

            trend = temporal.groupby("period").agg(
                reviews=("user_id", "size"),
                unique_reviewers=("user_id", "nunique"),
                average_rating=("rating", "mean"),
            ).sort_index()

            c1, c2, c3 = st.columns(3)
            c1.metric("Reviews with valid dates", f"{len(temporal):,}")
            c2.metric("Time periods", f"{len(trend):,}")
            c3.metric("Unique reviewers", f"{temporal['user_id'].nunique():,}")

            st.markdown("### Review volume over time")
            st.area_chart(trend["reviews"])
            st.markdown("### Active reviewers over time")
            st.line_chart(trend["unique_reviewers"])
            st.markdown("### Average rating over time")
            st.line_chart(trend["average_rating"])

            st.markdown("### Saved model predictions over time")
            lookup_path = RESULTS / "review_lookup_test.csv"
            if lookup_path.exists():
                @st.cache_data(show_spinner="Loading held-out risk trends...")
                def load_temporal_predictions(path):
                    return pd.read_csv(
                        path,
                        usecols=[
                            "date_feature", "semantic_probability",
                            "semantic_prediction",
                        ],
                        low_memory=False,
                    )

                predictions = load_temporal_predictions(str(lookup_path))
                predictions["date"] = pd.to_datetime(
                    predictions["date_feature"], errors="coerce"
                )
                predictions = predictions.dropna(subset=["date"]).copy()
                if granularity == "Monthly":
                    predictions["period"] = predictions["date"].dt.to_period("M").astype(str)
                elif granularity == "Quarterly":
                    predictions["period"] = predictions["date"].dt.to_period("Q").astype(str)
                else:
                    predictions["period"] = predictions["date"].dt.to_period("Y").astype(str)

                risk_trend = predictions.groupby("period").agg(
                    mean_estimated_risk=("semantic_probability", "mean"),
                    model_flagged=("semantic_prediction", "sum"),
                    test_reviews=("semantic_prediction", "size"),
                ).sort_index()

                st.line_chart(risk_trend["mean_estimated_risk"])
                st.bar_chart(risk_trend["model_flagged"])
                with st.expander("View temporal data tables"):
                    st.dataframe(trend.reset_index(), width="stretch", hide_index=True)
                    st.dataframe(risk_trend.reset_index(), width="stretch", hide_index=True)
                st.caption(
                    "Risk and flagged-review trends use the held-out test subset, "
                    "not predictions for every review in the full dataset."
                )
            else:
                st.dataframe(trend.reset_index(), width="stretch", hide_index=True)
                st.info("Test prediction lookup is unavailable; risk trends are omitted.")
'''

path.write_text(source[:start] + replacement + source[end:], encoding="utf-8")
print("Updated Reviewer Analysis and Temporal Analysis.")
