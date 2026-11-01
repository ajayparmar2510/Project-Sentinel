from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

start = 'elif section == "Review Analyzer":'
end = 'elif section == "Model Comparison":'

if start not in source or end not in source:
    raise SystemExit("Could not find the Review Analyzer boundaries. No changes made.")

before, remainder = source.split(start, 1)
old_section, after = remainder.split(end, 1)

new_section = '''elif section == "Review Analyzer":
    st.header("Review Risk Analyzer")
    st.caption(
        "Explore held-out historical reviews, their saved model predictions, "
        "and the behavioral evidence available for each case."
    )

    lookup_path = RESULTS / "review_lookup_test.csv"

    if not lookup_path.exists():
        st.error("Review lookup data was not found. Run prepare_review_lookup.py first.")
    else:
        @st.cache_data(show_spinner="Loading review predictions...")
        def load_review_lookup(path):
            return pd.read_csv(path, low_memory=False)

        review_data = load_review_lookup(str(lookup_path))

        if review_data.empty:
            st.warning("The review lookup file contains no rows.")
        else:
            risk_order = review_data.sort_values(
                "semantic_probability", ascending=False
            )

            choice = st.selectbox(
                "Choose a test review",
                options=risk_order["row_id"].astype(int).tolist(),
                format_func=lambda rid: (
                    f"Review #{rid} · estimated risk "
                    f"{risk_order.loc[risk_order['row_id'].eq(rid), 'semantic_probability'].iloc[0]:.1%}"
                ),
            )

            review = review_data.loc[
                review_data["row_id"].eq(choice)
            ].iloc[0]

            score = float(review["semantic_probability"])
            prediction = int(review["semantic_prediction"])
            threshold = 0.20

            st.markdown("### Model result")
            left, right, third = st.columns(3)
            left.metric("Estimated suspiciousness", f"{score:.1%}")
            right.metric(
                "Model classification",
                "Flag for review" if prediction == 1 else "Lower estimated risk",
            )
            third.metric("Reference rating", f"{review['rating']:.0f}/5")

            st.progress(max(0.0, min(1.0, score)))
            st.caption(
                f"Saved semantic-model prediction at the project's "
                f"evaluation threshold of {threshold:.0%}. This is a model score, "
                "not a verified probability of fraud or proof that a review is fake."
            )

            st.markdown("### Original review")
            st.markdown(
                '<div class="sentinel-panel">'
                + str(review["review_text"])
                + "</div>",
                unsafe_allow_html=True,
            )

            st.markdown("### Evidence for review")
            st.caption(
                "These are observed engineered features for this case, not "
                "individual causal explanations of the model score."
            )

            a, b, c = st.columns(3)
            a.metric("Prior reviewer reviews", f"{int(review['user_prior_reviews']):,}")
            b.metric("Prior product reviews", f"{int(review['product_prior_reviews']):,}")
            c.metric(
                "Reviews by this user in 30 days",
                f"{int(review['user_prior_reviews_30d']):,}",
            )

            d, e, f = st.columns(3)
            d.metric(
                "Days since user's previous review",
                "Unavailable" if pd.isna(review["days_since_user_previous_review"])
                else f"{float(review['days_since_user_previous_review']):.1f}",
            )
            e.metric(
                "Text marked duplicate",
                "Yes" if int(review["review_text_is_duplicate"]) == 1 else "No",
            )
            f.metric(
                "Text reuse ratio",
                f"{float(review['review_text_reuse_ratio']):.1%}",
            )

            g, h, i = st.columns(3)
            g.metric(
                "Reviewer rating deviation",
                f"{float(review['user_rating_deviation']):+.2f}",
            )
            h.metric(
                "Product rating deviation",
                f"{float(review['product_rating_deviation']):+.2f}",
            )
            i.metric(
                "Repeated-word ratio",
                f"{float(review['repeated_word_ratio']):.1%}",
            )

            with st.expander("Show additional case details"):
                st.write("Review row ID:", int(review["row_id"]))
                st.write("Reviewer ID:", review["user_id_feature"])
                st.write("Product ID:", review["prod_id_feature"])
                st.write(
                    "Dataset reference label:",
                    "Fake-labeled" if int(review["fake_label_feature"]) == 1
                    else "Genuine-labeled",
                )
                st.caption(
                    "The reference label comes from the dataset and is not "
                    "independent confirmation of deceptive intent."
                )

            st.info(
                "Interpretation: use the score to prioritize human review. "
                "The displayed signals provide context; they do not, by "
                "themselves, establish that a review is deceptive."
            )

'''

path.write_text(before + new_section + end + after, encoding="utf-8")
print("Updated Review Analyzer section in dashboard.py")
