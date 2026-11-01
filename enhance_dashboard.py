from pathlib import Path

p = Path("dashboard.py")
s = p.read_text(encoding="utf-8")

if "elif section == \"Reviewer Analysis\":" in s:
    raise SystemExit("Reviewer Analysis already exists; no changes made.")

review_start = s.find('            choices = risk_order.head(100).copy()')
review_end = s.find('            score = float(review["semantic_probability"])', review_start)

if review_start == -1 or review_end == -1:
    raise SystemExit("Could not find the current review selector. No changes made.")

selector = '''            valid_ids = sorted(review_data["row_id"].dropna().astype(int).unique())
            selected_id = st.selectbox(
                "Search all valid test review IDs",
                options=valid_ids,
                index=valid_ids.index(int(risk_order.iloc[0]["row_id"])),
                format_func=lambda rid: f"Review #{rid}",
            )
            review = review_data.loc[
                review_data["row_id"].eq(int(selected_id))
            ].iloc[0]

'''
s = s[:review_start] + selector + s[review_end:]

marker = 'elif section == "Model Comparison":'
if marker not in s:
    raise SystemExit("Model Comparison section not found. No changes made.")

pages = '''elif section == "Reviewer Analysis":
    st.header("Reviewer Behavior Analysis")
    st.caption("Explore reviewer activity using the held-out review lookup data.")
    lookup_path = RESULTS / "review_lookup_test.csv"
    if not lookup_path.exists():
        st.error("Review lookup file is missing.")
    else:
        @st.cache_data(show_spinner="Loading reviewer data...")
        def load_reviewer_data(path):
            return pd.read_csv(path, low_memory=False)
        df = load_reviewer_data(str(lookup_path))
        df["user_id_feature"] = df["user_id_feature"].astype(str)
        users = sorted(df["user_id_feature"].dropna().unique())
        user_id = st.selectbox("Select reviewer ID", users)
        user_df = df[df["user_id_feature"] == user_id].copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Reviews in test set", f"{len(user_df):,}")
        c2.metric("Mean estimated risk", f"{user_df['semantic_probability'].mean():.1%}")
        c3.metric("Flagged by model", f"{int((user_df['semantic_prediction'] == 1).sum()):,}")
        st.markdown("### Review activity over time")
        user_df["date_parsed"] = pd.to_datetime(user_df["date_feature"], errors="coerce")
        timeline = user_df.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("MS").size()
        if not timeline.empty:
            st.line_chart(timeline.rename("Reviews per month"))
        else:
            st.info("No usable dates for this reviewer.")
        st.markdown("### Individual reviews")
        show_cols = ["row_id", "date_feature", "rating", "semantic_probability",
                     "semantic_prediction", "user_burstiness_flag",
                     "review_text_is_duplicate", "review_text"]
        st.dataframe(user_df[show_cols].sort_values("semantic_probability", ascending=False),
                     width="stretch", hide_index=True)

elif section == "Temporal Analysis":
    st.header("Temporal Review Analysis")
    st.caption("Explore review volume and saved model risk scores over time.")
    lookup_path = RESULTS / "review_lookup_test.csv"
    if not lookup_path.exists():
        st.error("Review lookup file is missing.")
    else:
        @st.cache_data(show_spinner="Loading temporal data...")
        def load_temporal_data(path):
            return pd.read_csv(path, low_memory=False)
        df = load_temporal_data(str(lookup_path))
        df["date_parsed"] = pd.to_datetime(df["date_feature"], errors="coerce")
        df = df.dropna(subset=["date_parsed"]).copy()
        if df.empty:
            st.warning("No valid review dates were found.")
        else:
            granularity = st.radio("Time grouping", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
            if granularity == "Monthly":
                df["period"] = df["date_parsed"].dt.to_period("M").astype(str)
            elif granularity == "Quarterly":
                df["period"] = df["date_parsed"].dt.to_period("Q").astype(str)
            else:
                df["period"] = df["date_parsed"].dt.to_period("Y").astype(str)
            trend = df.groupby("period").agg(
                reviews=("row_id", "count"),
                mean_risk=("semantic_probability", "mean"),
                flagged=("semantic_prediction", "sum"),
                reference_fake_rate=("fake_label_feature", "mean"),
            ).sort_index()
            c1, c2, c3 = st.columns(3)
            c1.metric("Reviews with valid dates", f"{len(df):,}")
            c2.metric("Periods", f"{len(trend):,}")
            c3.metric("Mean estimated risk", f"{df['semantic_probability'].mean():.1%}")
            st.markdown("### Review volume")
            st.bar_chart(trend["reviews"])
            st.markdown("### Mean model risk over time")
            st.line_chart(trend["mean_risk"])
            st.markdown("### Model-flagged reviews over time")
            st.bar_chart(trend["flagged"])
            st.dataframe(trend.reset_index(), width="stretch", hide_index=True)
            st.caption("Reference fake-label rate is based on dataset labels, not independent confirmation of deception.")

'''

s = s.replace(marker, pages + marker, 1)

needle = '''            st.info(
                "Interpretation: use the score to prioritize human review. "
                "The displayed signals provide context; they do not, by "
                "themselves, establish that a review is deceptive."
            )
'''
replacement = '''            st.markdown("### Review-specific evidence summary")
            evidence = []
            if int(review["review_text_is_duplicate"]) == 1:
                evidence.append("The dataset marks this review text as a duplicate.")
            if float(review["review_text_reuse_ratio"]) > 0:
                evidence.append(f"Measured text reuse ratio: {float(review['review_text_reuse_ratio']):.1%}.")
            if int(review["user_burstiness_flag"]) == 1:
                evidence.append("The reviewer has a recorded activity-burstiness flag.")
            if float(review["user_rating_deviation"]) >= 1.5:
                evidence.append("The rating is notably above the reviewer's historical average.")
            elif float(review["user_rating_deviation"]) <= -1.5:
                evidence.append("The rating is notably below the reviewer's historical average.")
            if float(review["product_rating_deviation"]) >= 1.5:
                evidence.append("The rating is notably above the product's historical average.")
            elif float(review["product_rating_deviation"]) <= -1.5:
                evidence.append("The rating is notably below the product's historical average.")
            if float(review["repeated_word_ratio"]) > 0.2:
                evidence.append("The text has a comparatively high repeated-word ratio.")
            if evidence:
                for item in evidence:
                    st.markdown(f"- {item}")
            else:
                st.write("No listed signal crossed the simple display rules for this case. This does not establish that the review is genuine.")
            st.caption("These are rule-based summaries of observed features, not causal explanations of the model score.")

            st.info(
                "Interpretation: use the score to prioritize human review. "
                "The displayed signals provide context; they do not, by "
                "themselves, establish that a review is deceptive."
            )
'''
if needle not in s:
    raise SystemExit("Could not locate the review interpretation block. No changes made.")
s = s.replace(needle, replacement, 1)

p.write_text(s, encoding="utf-8")
print("Added Reviewer Analysis, Temporal Analysis, all-ID selection, and review evidence summary.")
