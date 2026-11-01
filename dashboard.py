import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATASET_PATH = ROOT / "results" / "phase5_engineered_features.csv.gz"
EXPLAIN = ROOT / "explainability"

st.set_page_config(
    page_title="Project Sentinel | Trust Intelligence",
    page_icon="🛡️",
    layout="wide",
)

# Shared investigation context across dashboard sections.
if "sentinel_selected_review_id" not in st.session_state:
    st.session_state["sentinel_selected_review_id"] = None

if "sentinel_selected_reviewer_id" not in st.session_state:
    st.session_state["sentinel_selected_reviewer_id"] = None

if "sentinel_selected_product_id" not in st.session_state:
    st.session_state["sentinel_selected_product_id"] = None


def load_json(filename):
    path = RESULTS / filename
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)

@st.cache_data(show_spinner="Loading YelpZip historical data...")
def load_historical_data():
    columns = [
        "row_id", "user_id", "prod_id", "rating", "fake_label", "date",
        "review_year", "review_month", "review_day_of_week",
        "user_prior_reviews", "user_prior_unique_products",
        "user_prior_avg_rating", "user_prior_avg_review_length",
        "days_since_user_previous_review", "user_prior_reviews_7d",
        "user_prior_reviews_30d", "user_prior_reviews_90d",
        "user_prior_reviews_365d", "user_burstiness_flag",
        "user_rating_deviation", "user_product_diversity", "user_rating_std",
        "product_prior_reviews", "product_prior_avg_rating",
        "product_rating_deviation", "review_text_is_duplicate",
        "review_text_reuse_ratio", "repeated_word_ratio"
    ]
    if not DATASET_PATH.exists():
        return pd.DataFrame()
    available = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()
    selected = [column for column in columns if column in available]
    data = pd.read_csv(DATASET_PATH, usecols=selected, low_memory=False)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["rating", "fake_label"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def load_csv(filename):
    path = RESULTS / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def pct(value):
    return f"{float(value):.2%}"

def show_metric_pair(df, metric, label, format_as_pct=True):
    if df.empty or metric not in df.columns or "model" not in df.columns:
        st.info(f"{label} is not available in the saved metrics file.")
        return
    values = df.set_index("model")[metric]
    cols = st.columns(2)
    for col, name in zip(cols, ["baseline", "semantic"]):
        if name in values.index:
            value = values.loc[name]
            display = pct(value) if format_as_pct else f"{value:.4f}"
            col.metric(name.title(), display, label)

css_path = ROOT / "assets" / "sentinel.css"
if css_path.exists():
    st.markdown("<style>" + css_path.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

st.markdown("""
<div class="sentinel-hero">
  <div class="sentinel-eyebrow">TRUST INTELLIGENCE · RESEARCH PROTOTYPE</div>
  <div class="sentinel-hero-title">Project Sentinel</div>
  <div class="sentinel-hero-copy">An explainable framework for analyzing review credibility, reviewer behavior, product patterns, and model-estimated risk.</div>
</div>
""", unsafe_allow_html=True)
st.subheader("Explainable Trust Intelligence Framework for Fake Review Detection")
st.caption(
    "Research dashboard | YelpZip dataset | Saved experiment artifacts | "
    "Model predictions are not definitive fraud verdicts."
)

metrics = load_csv("phase19_overall_test_metrics.csv")
comparison = load_csv("phase16_model_comparison_metrics.csv")
trust = load_json("phase17_trust_intelligence_scores_summary.json")
calibration = load_json("phase17_validation_calibration_summary.json")
errors = load_json("phase19_error_analysis_summary.json")

if metrics.empty:
    st.error("The Phase 19 metrics file could not be loaded. Check the results folder.")
    st.stop()

baseline = metrics[metrics["model"].astype(str).str.lower() == "baseline"]
semantic = metrics[metrics["model"].astype(str).str.lower() == "semantic"]

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select a section",
    [
        "Overview",
        "Review Analyzer",
        "Temporal Analysis",
        "Reviewer Analysis",
        "Model Comparison",
        "Error Analysis",
        "Trust Intelligence",
        "Explainability",
        "Project Notes",
    ],
)
st.sidebar.caption("Metrics are read from saved project files.")

# Header-level KPIs
total_test = int(metrics["test_rows"].iloc[0])
prevalence = float(metrics["fake_prevalence"].iloc[0])
st.caption(f"Evaluation threshold: {float(metrics['threshold'].iloc[0]):.2f}")

if section == "Overview":
    st.header("Model Performance Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Held-out test reviews", f"{total_test:,}")
    c2.metric("Reference fake-label prevalence", pct(prevalence))
    if not baseline.empty:
        c3.metric("Baseline F1", pct(baseline["f1"].iloc[0]))
    if not semantic.empty:
        c4.metric("Semantic F1", pct(semantic["f1"].iloc[0]))

    st.markdown("### Model comparison")
    if not baseline.empty and not semantic.empty:
        overview = metrics.set_index("model")[
            ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
        ].T
        overview.index = [
            "Precision", "Recall", "F1", "Accuracy", "ROC-AUC", "PR-AUC"
        ]
        chart_data = overview.rename_axis("Metric").reset_index().melt(
            id_vars="Metric",
            var_name="Model",
            value_name="Score",
        )
        fig = px.bar(
            chart_data,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text=chart_data["Score"].map(lambda value: f"{value:.1%}"),
            color_discrete_map={
                "baseline": "#8AA4D6",
                "semantic": "#356AE6",
            },
            template="plotly_white",
            title="Baseline vs Semantic Model",
            labels={"Score": "Evaluation score", "Metric": "Metric"},
        )
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{fullData.name}: %{y:.2%}<extra></extra>",
        )
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Arial, sans-serif", color="#17233D", size=13),
            title_font=dict(size=19, color="#17233D"),
            legend_title_text="Model",
            margin=dict(l=20, r=20, t=70, b=25),
            yaxis=dict(
                title="Score",
                tickformat=".0%",
                rangemode="tozero",
                gridcolor="#E8EDF5",
            ),
            xaxis=dict(title="", showgrid=False),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            metrics[
                [
                    "model", "threshold", "test_rows", "precision", "recall",
                    "f1", "accuracy", "roc_auc", "pr_auc"
                ]
            ].set_index("model").style.format({
                "precision": "{:.2%}",
                "recall": "{:.2%}",
                "f1": "{:.2%}",
                "accuracy": "{:.2%}",
                "roc_auc": "{:.4f}",
                "pr_auc": "{:.4f}",
            }),
            width="stretch",
        )

    st.info(
        "Interpretation: the semantic model improves recall and F1 modestly, "
        "but its false-positive count is higher. Choose a model based on the "
        "intended review-screening use case and the cost of each error."
    )

elif section == "Review Analyzer":
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
            threshold = 0.20
            st.markdown("### Find a review")
            st.caption(
                "Choose from the highest-risk saved test cases. Internal dataset IDs "
                "are kept out of the visible selector."
            )

            # Limit visible options and compute labels once instead of scanning
            # the complete dataset separately for every dropdown option.
            candidates = (
                review_data.dropna(subset=["row_id", "semantic_probability"])
                .sort_values("semantic_probability", ascending=False)
                
                .copy()
                .reset_index(drop=True)
            )

            if candidates.empty:
                st.warning("No valid review cases are available.")
                st.stop()

            def risk_band(risk):
                if risk >= 0.60:
                    return "Elevated estimated risk"
                if risk >= threshold:
                    return "Flagged for review"
                return "Lower estimated risk"

            candidates["row_id"] = candidates["row_id"].astype(int)

            reviewer_filter = st.text_input(
                "Filter by reviewer ID (optional)",
                placeholder="For example: 8301",
                key="risk_analyzer_reviewer_filter",
            ).strip()

            if reviewer_filter:
                candidates = candidates[
                    candidates["user_id_feature"].astype("string").str.contains(
                        reviewer_filter, regex=False, na=False
                    )
                ].copy()

            if candidates.empty:
                st.warning("No saved prediction cases match this reviewer ID.")
                st.stop()
            candidates["case_label"] = candidates.apply(
                lambda row: (
                    f"Reviewer ID: {row.get('user_id_feature', 'Unavailable')} "
                    f"· Review {int(row['row_id'])}"
                ),
                axis=1,
            )

            label_by_id = dict(zip(candidates["row_id"], candidates["case_label"]))
            candidate_ids = candidates["row_id"].tolist()

            selected_id = st.selectbox(
                "Review case",
                options=candidate_ids,
                index=0,
                format_func=lambda rid: label_by_id[int(rid)],
                help="Showing the 200 highest-risk saved test cases. Risk is a model estimate used to prioritize human review, not proof of deception.",
            )

            review = candidates.loc[
                candidates["row_id"].eq(int(selected_id))
            ].iloc[0]

            # Keep the selected case connected across dashboard sections.
            st.session_state["sentinel_selected_review_id"] = int(selected_id)

            reviewer_key = review.get("user_id_feature")
            product_key = review.get("prod_id_feature")

            st.session_state["sentinel_selected_reviewer_id"] = (
                str(reviewer_key)
                if pd.notna(reviewer_key) else None
            )
            st.session_state["sentinel_selected_product_id"] = (
                str(product_key)
                if pd.notna(product_key) else None
            )

            score = float(review["semantic_probability"])
            prediction = int(review["semantic_prediction"])

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

            st.markdown("### Evidence & interpretation")
            st.caption(
                "These cards explain measurable patterns in this review and its "
                "history. They are screening clues, not proof of deceptive intent."
            )

            evidence_cards = []

            duplicate_flag = pd.to_numeric(
                pd.Series([review.get("review_text_is_duplicate")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(duplicate_flag) and int(duplicate_flag) == 1:
                evidence_cards.append((
                    "Text duplication detected",
                    "The dataset marks this review text as a duplicate.",
                    "Repeated text can be worth checking across reviews, but "
                    "templates, quotations, or legitimate repeated wording can "
                    "also create matches.",
                    "Text pattern",
                ))

            reuse = pd.to_numeric(
                pd.Series([review.get("review_text_reuse_ratio")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(reuse) and reuse > 0:
                evidence_cards.append((
                    "Text reuse",
                    f"The measured text reuse ratio is {reuse:.1%}.",
                    "This feature indicates text reuse under the dataset's "
                    "definition. It does not establish who copied the text or why.",
                    "Language signal",
                ))

            burst = pd.to_numeric(
                pd.Series([review.get("user_burstiness_flag")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(burst) and int(burst) == 1:
                evidence_cards.append((
                    "Reviewer activity burst",
                    "A burstiness flag is recorded for this reviewer.",
                    "Concentrated activity may deserve context checks. Promotions, "
                    "busy periods, or normal user behavior may also explain bursts.",
                    "Behavioral signal",
                ))

            user_dev = pd.to_numeric(
                pd.Series([review.get("user_rating_deviation")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(user_dev) and abs(user_dev) >= 1.5:
                direction = "above" if user_dev > 0 else "below"
                evidence_cards.append((
                    "Deviation from reviewer history",
                    f"This rating is {abs(user_dev):.2f} points {direction} "
                    "the reviewer's historical average, according to the saved feature.",
                    "A change from someone's usual ratings can be informative, "
                    "but people can genuinely have unusually positive or negative experiences.",
                    "Rating context",
                ))

            product_dev = pd.to_numeric(
                pd.Series([review.get("product_rating_deviation")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(product_dev) and abs(product_dev) >= 1.5:
                direction = "above" if product_dev > 0 else "below"
                evidence_cards.append((
                    "Deviation from product history",
                    f"This rating is {abs(product_dev):.2f} points {direction} "
                    "the product's historical average, according to the saved feature.",
                    "An unusual rating may reflect a genuine outlier experience. "
                    "Consider the review text and other context before interpreting it.",
                    "Product context",
                ))

            repeated = pd.to_numeric(
                pd.Series([review.get("repeated_word_ratio")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(repeated) and repeated > 0.2:
                evidence_cards.append((
                    "Repeated wording",
                    f"The repeated-word ratio is {repeated:.1%}, above the "
                    "dashboard's simple 20% display threshold.",
                    "Repetition may reflect writing style, short text, or emphasis. "
                    "It is a descriptive feature, not a standalone deception detector.",
                    "Language signal",
                ))

            if evidence_cards:
                for start in range(0, len(evidence_cards), 2):
                    columns = st.columns(2)
                    for column, card in zip(columns, evidence_cards[start:start + 2]):
                        title, observed, interpretation, category = card
                        with column:
                            st.markdown(
                                '<div class="sentinel-insight">'
                                f'<div class="sentinel-insight-title">{title}</div>'
                                f'<div style="color:#65DCCB;font-size:.75rem;'
                                f'font-weight:750;letter-spacing:.08em;'
                                f'text-transform:uppercase;margin-bottom:.45rem;">'
                                f'{category}</div>'
                                f'<div>{observed}</div>'
                                f'<div style="margin-top:.65rem;color:#A0B0CA;">'
                                f'<b>How to interpret:</b> {interpretation}</div>'
                                '</div>',
                                unsafe_allow_html=True,
                            )
            else:
                st.markdown(
                    '<div class="sentinel-insight">'
                    '<div class="sentinel-insight-title">No prominent rule-based signals</div>'
                    'None of the dashboard display rules was triggered for this case. '
                    'This does not mean the review is genuine; it means these selected '
                    'features did not cross their display thresholds.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### Analyst takeaway")
            if score >= threshold:
                st.markdown(
                    '<div class="sentinel-insight">'
                    '<div class="sentinel-insight-title">Prioritize for human review</div>'
                    f'The saved model score is <b>{score:.1%}</b>, at or above '
                    f'the configured {threshold:.0%} threshold. '
                    'Use the evidence above to guide further checks; the score is '
                    'not a verified probability of fraud.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="sentinel-insight">'
                    '<div class="sentinel-insight-title">Lower model-estimated risk</div>'
                    f'The saved model score is <b>{score:.1%}</b>, below the '
                    f'configured {threshold:.0%} threshold. This is not a guarantee '
                    'of authenticity; consider the available context and evidence.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                "The model score and rule-based evidence cards are different views: "
                "the cards summarize selected observed features and do not explain "
                "the model's internal decision causally."
            )

elif section == "Reviewer Analysis":
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
            eligible = eligible.reset_index(drop=True)
            eligible["selector_label"] = eligible["user_id"].astype(str)
            st.caption(
                "Reviewer names are anonymized in this selector. Review counts "
                "describe activity and are not evidence of deceptive behavior."
            )
            shared_reviewer = st.session_state.get(
                "sentinel_selected_reviewer_id"
            )

            # Preselect the reviewer linked to the chosen review, when eligible.
            default_selector_index = next(
                (
                    i for i, user_id in enumerate(eligible["user_id"])
                    if shared_reviewer is not None
                    and str(user_id) == str(shared_reviewer)
                ),
                0,
            )

            selected_label = st.selectbox(
                "Select reviewer",
                eligible["selector_label"].tolist(),
                index=default_selector_index,
            )
            selected_index = eligible.index[
                eligible["selector_label"].eq(selected_label)
            ][0]
            selected_user = str(eligible.loc[selected_index, "user_id"])

            # Remember manual reviewer selections for other sections.
            st.session_state["sentinel_selected_reviewer_id"] = selected_user

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
            st.caption(
                "This chart counts the selected reviewer's dated reviews by month. "
                "It helps distinguish steady activity from periods with higher volume."
            )
            if not history.empty and history["date"].notna().any():
                monthly = (
                    history.dropna(subset=["date"])
                    .set_index("date").resample("MS").size()
                    .rename("Reviews")
                )
                monthly_df = monthly.rename_axis("Month").reset_index()
                fig = px.area(
                    monthly_df,
                    x="Month",
                    y="Reviews",
                    template="plotly_dark",
                    title="Monthly review activity",
                    labels={"Month": "Month", "Reviews": "Number of reviews"},
                )
                fig.update_traces(
                    line=dict(color="#6595FF", width=2),
                    fillcolor="rgba(101,149,255,0.20)",
                    hovertemplate="%{x|%b %Y}<br>Reviews: %{y}<extra></extra>",
                )
                fig.update_layout(
                    paper_bgcolor="#101827",
                    plot_bgcolor="#101827",
                    font=dict(color="#F0F5FF"),
                    title_font=dict(color="#F0F5FF", size=18),
                    margin=dict(l=15, r=15, t=55, b=15),
                    xaxis=dict(
                        title="Month",
                        type="date",
                        showgrid=False,
                    ),
                    yaxis=dict(
                        title="Number of reviews",
                        rangemode="tozero",
                        gridcolor="#263650",
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
                if not monthly.empty:
                    peak_month = monthly.idxmax()
                    peak_count = int(monthly.max())
                    active_months = int((monthly > 0).sum())
                    st.markdown(
                        '<div class="sentinel-insight">'
                        '<div class="sentinel-insight-title">Timeline insight</div>'
                        f'The highest-volume month was <b>{peak_month:%B %Y}</b>, '
                        f'with <b>{peak_count:,} reviews</b>. '
                        f'The reviewer had recorded activity in {active_months:,} '
                        'months represented by the timeline. Check surrounding context '
                        'before interpreting a spike.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                st.caption(
                    "A concentrated period may merit closer inspection. "
                    "Activity alone does not establish coordinated or fake reviews."
                )
            else:
                st.info("No valid dates were available for this reviewer.")

            st.markdown("### Rating distribution")
            st.caption(
                "This chart shows how often the reviewer gave each rating. "
                "The distribution provides context for the average rating above."
            )
            if "rating" in history.columns:
                ratings = history["rating"].value_counts().sort_index()
                st.bar_chart(ratings.rename_axis("Rating").rename("Reviews"))
                valid_ratings = pd.to_numeric(
                    history["rating"], errors="coerce"
                ).dropna()
                if not valid_ratings.empty:
                    dominant_rating = int(valid_ratings.mode().iloc[0])
                    positive_share = float((valid_ratings >= 4).mean())
                    st.markdown(
                        '<div class="sentinel-insight">'
                        '<div class="sentinel-insight-title">Rating insight</div>'
                        f'The most common rating was <b>{dominant_rating}/5</b>. '
                        f'<b>{positive_share:.1%}</b> of valid ratings were 4 or 5. '
                        'A positive or negative rating pattern is not, by itself, '
                        'evidence that reviews are inauthentic.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

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
                    individual["row_id"] = pd.to_numeric(
                        individual["row_id"], errors="coerce"
                    )
                    individual["semantic_probability"] = pd.to_numeric(
                        individual["semantic_probability"], errors="coerce"
                    )
                    individual = individual.dropna(
                        subset=["row_id", "semantic_probability"]
                    ).copy()

                    if individual.empty:
                        st.info("No valid saved review predictions are available for this reviewer.")
                    else:
                        individual["row_id"] = individual["row_id"].astype(int)
                        individual = individual.sort_values(
                            "semantic_probability", ascending=False
                        ).reset_index(drop=True)

                        st.markdown("#### Select an individual review")
                        st.caption(
                            "Choose a saved held-out review to connect it to the "
                            "Trust Intelligence and Temporal Analysis sections."
                        )

                        review_options = individual["row_id"].tolist()
                        shared_review = st.session_state.get(
                            "sentinel_selected_review_id"
                        )

                        default_review_index = (
                            review_options.index(int(shared_review))
                            if shared_review is not None
                            and int(shared_review) in review_options
                            else 0
                        )

                        selected_review_id = st.selectbox(
                            "Individual review",
                            options=review_options,
                            index=default_review_index,
                            format_func=lambda rid: (
                                f"Review {rid} · "
                                f"Risk {float(individual.loc[individual['row_id'].eq(rid), 'semantic_probability'].iloc[0]):.1%}"
                            ),
                            key="reviewer_analysis_review_selector",
                        )

                        selected_case = individual.loc[
                            individual["row_id"].eq(int(selected_review_id))
                        ].iloc[0]

                        st.session_state["sentinel_selected_review_id"] = int(
                            selected_case["row_id"]
                        )
                        st.session_state["sentinel_selected_reviewer_id"] = str(
                            selected_case["user_id_feature"]
                        )
                        selected_product = selected_case.get("prod_id_feature")
                        st.session_state["sentinel_selected_product_id"] = (
                            str(selected_product)
                            if pd.notna(selected_product)
                            else None
                        )

                        case_a, case_b, case_c = st.columns(3)
                        case_a.metric(
                            "Selected review risk",
                            f"{float(selected_case['semantic_probability']):.1%}",
                        )
                        case_b.metric(
                            "Model result",
                            "Flagged for review"
                            if int(selected_case["semantic_prediction"]) == 1
                            else "Below threshold",
                        )
                        case_c.metric(
                            "Rating",
                            f"{float(selected_case['rating']):.1f}/5"
                            if pd.notna(selected_case.get("rating"))
                            else "Unavailable",
                        )

                        st.markdown("#### Selected review text")
                        review_text = selected_case.get("review_text")
                        if pd.notna(review_text) and str(review_text).strip():
                            st.text(str(review_text))
                        else:
                            st.info("Review text is unavailable for this record.")

                        evidence_columns = [
                            ("user_prior_reviews", "Prior reviewer reviews"),
                            ("user_prior_reviews_30d", "Reviewer reviews in prior 30 days"),
                            ("days_since_user_previous_review", "Days since previous review"),
                            ("user_burstiness_flag", "Activity burst flag"),
                            ("review_text_is_duplicate", "Duplicate-text flag"),
                            ("review_text_reuse_ratio", "Text reuse ratio"),
                            ("user_rating_deviation", "Reviewer rating deviation"),
                            ("product_rating_deviation", "Product rating deviation"),
                            ("repeated_word_ratio", "Repeated-word ratio"),
                        ]
                        # Retrieve the selected review's prior-review count
                        # from the complete engineered dataset, not the test lookup.
                        historical = load_historical_data()
                        full_case = pd.DataFrame()
                        if not historical.empty and "row_id" in historical.columns:
                            full_case = historical.loc[
                                pd.to_numeric(historical["row_id"], errors="coerce").eq(
                                    int(selected_case["row_id"])
                                )
                            ]

                        evidence_rows = []
                        for column_name, display_name in evidence_columns:
                            if column_name == "user_prior_reviews":
                                if (
                                    not full_case.empty
                                    and column_name in full_case.columns
                                ):
                                    raw_value = full_case.iloc[0][column_name]
                                else:
                                    raw_value = None
                            elif column_name in selected_case.index:
                                raw_value = selected_case[column_name]
                            else:
                                continue

                            value = pd.to_numeric(
                                pd.Series([raw_value]),
                                errors="coerce",
                            ).iloc[0]
                            if pd.isna(value):
                                shown_value = "Unavailable"
                            elif column_name in {
                                "review_text_reuse_ratio", "repeated_word_ratio"
                            }:
                                shown_value = f"{value:.1%}"
                            elif column_name in {
                                "user_rating_deviation", "product_rating_deviation"
                            }:
                                shown_value = f"{value:+.2f}"
                            elif column_name == "days_since_user_previous_review":
                                shown_value = f"{value:.1f} days"
                            elif column_name in {
                                "user_burstiness_flag", "review_text_is_duplicate"
                            }:
                                shown_value = "Yes" if int(value) == 1 else "No"
                            else:
                                shown_value = f"{value:,.0f}"
                            evidence_rows.append({
                                "Observed signal": display_name,
                                "Saved value": shown_value,
                            })

                        if evidence_rows:
                            st.dataframe(
                                pd.DataFrame(evidence_rows),
                                width="stretch",
                                hide_index=True,
                            )

                        st.caption(
                            "These are saved model outputs and observed features. "
                            "They are investigation clues, not proof of fraud or "
                            "causal explanations of the model's decision."
                        )

                    flagged = int((individual["semantic_prediction"] == 1).sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Test-set reviews", f"{len(individual):,}")
                    c2.metric("Model-flagged test reviews", f"{flagged:,}")
                    c3.metric(
                        "Mean estimated risk",
                        f"{individual['semantic_probability'].mean():.1%}",
                    )

                    st.caption(
                        "The table is sorted from highest to lowest saved model score. "
                        "Internal row IDs are hidden from this view."
                    )

                    display_names = {
                        "date_feature": "Review date",
                        "rating": "Rating",
                        "semantic_probability": "Estimated risk score",
                        "semantic_prediction": "Model flag",
                        "user_burstiness_flag": "Activity burst flag",
                        "review_text_is_duplicate": "Duplicate-text flag",
                        "review_text_reuse_ratio": "Text reuse ratio",
                        "user_rating_deviation": "Reviewer rating deviation",
                        "product_rating_deviation": "Product rating deviation",
                        "review_text": "Review text",
                    }
                    show_cols = [
                        c for c in [
                            "date_feature", "rating",
                            "semantic_probability", "semantic_prediction",
                            "user_burstiness_flag", "review_text_is_duplicate",
                            "review_text_reuse_ratio", "user_rating_deviation",
                            "product_rating_deviation", "review_text",
                        ] if c in individual.columns
                    ]
                    display = individual[show_cols].sort_values(
                        "semantic_probability", ascending=False
                    ).rename(columns=display_names)

                    if "Estimated risk score" in display.columns:
                        display["Estimated risk score"] = display[
                            "Estimated risk score"
                        ].map(lambda value: f"{value:.1%}")
                    st.dataframe(
                        display,
                        width="stretch",
                        hide_index=True,
                    )

                    st.markdown("### How to interpret a flagged review")
                    st.markdown(
                        '<div class="sentinel-insight">'
                        '<div class="sentinel-insight-title">Use multiple signals</div>'
                        'The model score helps prioritize reviews for human inspection. '
                        'Activity bursts, duplicate-text indicators, text reuse, and '
                        'rating deviations add context but do not prove deception. '
                        'These are saved held-out predictions, not live inference.'
                        '</div>',
                        unsafe_allow_html=True,
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

            # Shared investigation: show the selected reviewer's activity
            # alongside the existing full-dataset temporal analysis.
            shared_reviewer = st.session_state.get(
                "sentinel_selected_reviewer_id"
            )

            if shared_reviewer is not None:
                reviewer_temporal = temporal[
                    temporal["user_id"].astype(str).eq(str(shared_reviewer))
                ].copy()

                st.markdown("### Connected investigation: selected reviewer")

                if reviewer_temporal.empty:
                    st.info(
                        "The selected reviewer has no valid dated records "
                        "in the temporal dataset."
                    )
                else:
                    st.caption(
                        "This view follows the reviewer selected in Review Analyzer "
                        "or Reviewer Intelligence. It shows dated activity, not proof "
                        "of deceptive behavior."
                    )

                    reviewer_periods = (
                        reviewer_temporal.groupby("period")
                        .size()
                        .rename("Reviews")
                        .rename_axis("Period")
                        .reset_index()
                    )

                    rc1, rc2 = st.columns(2)
                    rc1.metric(
                        "Dated reviews for selected reviewer",
                        f"{len(reviewer_temporal):,}",
                    )
                    rc2.metric(
                        "Periods with activity",
                        f"{len(reviewer_periods):,}",
                    )

                    reviewer_fig = px.bar(
                        reviewer_periods,
                        x="Period",
                        y="Reviews",
                        template="plotly_dark",
                        title="Selected reviewer activity by period",
                    )
                    reviewer_fig.update_traces(
                        marker_color="#6595FF",
                        hovertemplate="%{x}<br>Reviews: %{y}<extra></extra>",
                    )
                    reviewer_fig.update_layout(
                        xaxis_title="Time period",
                        yaxis_title="Number of dated reviews",
                    )
                    st.plotly_chart(reviewer_fig, use_container_width=True)

                    reviewer_mean = reviewer_temporal["rating"].mean()
                    if pd.notna(reviewer_mean):
                        st.caption(
                            f"Average rating across these dated records: "
                            f"{reviewer_mean:.2f}/5. Compare this with the "
                            "reviewer's profile and the overall rating trend."
                        )
            else:
                st.info(
                    "Choose a review in Review Analyzer or a reviewer in "
                    "Reviewer Intelligence to display connected reviewer activity here."
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Reviews with valid dates", f"{len(temporal):,}")
            c2.metric("Time periods", f"{len(trend):,}")
            c3.metric("Unique reviewers", f"{temporal['user_id'].nunique():,}")

            st.markdown("### Review volume over time")
            st.caption(
                "What it shows: the number of reviews recorded in each selected "
                "time period across the full dataset."
            )
            st.area_chart(trend["reviews"])

            peak_period = trend["reviews"].idxmax()
            quiet_period = trend["reviews"].idxmin()
            st.markdown(
                '<div class="sentinel-insight">'
                '<div class="sentinel-insight-title">How to interpret this trend</div>'
                f'The highest-volume period is <b>{peak_period}</b> '
                f'({int(trend.loc[peak_period, "reviews"]):,} reviews); '
                f'the lowest-volume period is <b>{quiet_period}</b> '
                f'({int(trend.loc[quiet_period, "reviews"]):,} reviews). '
                'Spikes can reflect genuine demand, seasonal effects, data coverage, '
                'or coordinated activity. This chart alone cannot distinguish them.'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown("### Active reviewers over time")
            st.caption(
                "What it shows: how many distinct reviewer accounts contributed "
                "at least one review during each period."
            )
            st.line_chart(trend["unique_reviewers"])

            st.markdown(
                '<div class="sentinel-insight">'
                '<div class="sentinel-insight-title">What this can reveal</div>'
                'Compare this line with review volume. If review counts rise faster '
                'than the number of active reviewers, activity per reviewer may '
                'also have increased. That is a useful lead for further investigation, '
                'not evidence of misconduct by itself.'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown("### Average rating over time")
            st.caption(
                "What it shows: the mean recorded rating for reviews in each period, "
                "on the dataset rating scale."
            )
            st.line_chart(trend["average_rating"])

            rating_valid = trend["average_rating"].dropna()
            if not rating_valid.empty:
                st.markdown(
                    '<div class="sentinel-insight">'
                    '<div class="sentinel-insight-title">Rating pattern</div>'
                    f'The period averages range from '
                    f'<b>{rating_valid.min():.2f}</b> to '
                    f'<b>{rating_valid.max():.2f}</b>. '
                    'Changes may reflect changing products, customers, or review '
                    'coverage. Compare ratings with volume and product context before '
                    'drawing conclusions about review credibility.'
                    '</div>',
                    unsafe_allow_html=True,
                )

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

                st.markdown("#### Average estimated risk by period")
                st.caption(
                    "This trend uses saved predictions for held-out test reviews only. "
                    "It shows the mean model score per period, not a verified fraud rate."
                )
                st.line_chart(risk_trend["mean_estimated_risk"])

                if not risk_trend.empty:
                    peak_risk_period = risk_trend["mean_estimated_risk"].idxmax()
                    st.markdown(
                        '<div class="sentinel-insight">'
                        '<div class="sentinel-insight-title">Risk trend insight</div>'
                        f'The highest average estimated risk occurs in '
                        f'<b>{peak_risk_period}</b> '
                        f'({risk_trend.loc[peak_risk_period, "mean_estimated_risk"]:.1%}). '
                        'Review the number of test examples in each period before '
                        'comparing periods; smaller groups can fluctuate more.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Reviews flagged by the model")
                st.caption(
                    "Counts saved model predictions marked for review at the configured "
                    "threshold. Counts can rise simply because more test reviews occur "
                    "in a period."
                )
                st.bar_chart(risk_trend["model_flagged"])

                st.markdown(
                    '<div class="sentinel-insight">'
                    '<div class="sentinel-insight-title">Important distinction</div>'
                    'A flagged count is not the same as a suspicious-review percentage. '
                    'Use the test-review count alongside it, and remember that these '
                    'predictions cover only the held-out test subset.'
                    '</div>',
                    unsafe_allow_html=True,
                )
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
elif section == "Model Comparison":
    st.header("Detailed Model Comparison")
    st.write("Held-out test split results at the saved threshold.")
    st.dataframe(metrics, width="stretch")

    if not comparison.empty:
        with st.expander("Additional saved training and validation details"):
            st.dataframe(comparison, width="stretch")

    st.markdown("### Confusion-matrix counts")
    for _, row in metrics.iterrows():
        st.markdown(f"**{str(row['model']).title()} model**")
        matrix = pd.DataFrame(
            [
                [int(row["true_negative"]), int(row["false_positive"])],
                [int(row["false_negative"]), int(row["true_positive"])],
            ],
            index=["Actual genuine", "Actual fake"],
            columns=["Predicted genuine", "Predicted fake"],
        )
        st.dataframe(matrix, width="stretch")

elif section == "Error Analysis":
    st.header("Error Analysis")
    st.caption(
        "These are disagreements with the dataset's reference labels, "
        "not independently confirmed fraud cases."
    )

    error_rows = []
    for _, row in metrics.iterrows():
        error_rows.append({
            "Model": str(row["model"]).title(),
            "False positives": int(row["false_positive"]),
            "False negatives": int(row["false_negative"]),
            "True positives": int(row["true_positive"]),
            "True negatives": int(row["true_negative"]),
        })
    error_df = pd.DataFrame(error_rows).set_index("Model")
    st.dataframe(error_df, width="stretch")
    st.bar_chart(error_df[["False positives", "False negatives"]])

    changed = RESULTS / "phase19_changed_prediction_summary.csv"
    if changed.exists():
        st.markdown("### Changed predictions between models")
        changed_df = pd.read_csv(changed)
        st.dataframe(changed_df, width="stretch")
        st.bar_chart(changed_df.set_index("change_type")["count"])

    if errors:
        with st.expander("Saved error-analysis audit details"):
            st.json(errors)

elif section == "Trust Intelligence":
    st.header("Trust Intelligence Score")

    # Contextual case panel: uses the same saved review selected elsewhere.
    st.markdown("### Connected investigation context")
    shared_review_id = st.session_state.get("sentinel_selected_review_id")
    lookup_path = RESULTS / "review_lookup_test.csv"

    if shared_review_id is None:
        st.info(
            "Select a case in Review Analyzer to inspect its saved model "
            "prediction and behavioral evidence here."
        )
    elif not lookup_path.exists():
        st.warning("The saved review lookup is unavailable.")
    else:
        @st.cache_data(show_spinner="Loading selected case context...")
        def load_trust_case_lookup(path):
            return pd.read_csv(path, low_memory=False)

        trust_cases = load_trust_case_lookup(str(lookup_path))
        if "row_id" in trust_cases.columns:
            matched = trust_cases[
                pd.to_numeric(trust_cases["row_id"], errors="coerce")
                == int(shared_review_id)
            ]

            if matched.empty:
                st.warning(
                    "The selected review ID was not found in the saved test lookup."
                )
            else:
                case = matched.iloc[0]
                risk = pd.to_numeric(
                    pd.Series([case.get("semantic_probability")]),
                    errors="coerce",
                ).iloc[0]
                prediction = pd.to_numeric(
                    pd.Series([case.get("semantic_prediction")]),
                    errors="coerce",
                ).iloc[0]

                case_a, case_b, case_c = st.columns(3)
                case_a.metric(
                    "Selected case",
                    "Available",
                )
                case_b.metric(
                    "Saved semantic risk",
                    f"{risk:.1%}" if pd.notna(risk) else "Unavailable",
                )
                case_c.metric(
                    "Saved model result",
                    (
                        "Flagged for review"
                        if int(prediction) == 1
                        else "Below threshold"
                    ) if pd.notna(prediction) else "Unavailable",
                )

                evidence_names = {
                    "user_prior_reviews": "Prior reviewer reviews",
                    "user_prior_reviews_30d": "Reviewer reviews in prior 30 days",
                    "product_prior_reviews": "Prior product reviews",
                    "days_since_user_previous_review": "Days since previous review",
                    "review_text_is_duplicate": "Text marked duplicate",
                    "review_text_reuse_ratio": "Text reuse ratio",
                    "user_rating_deviation": "Reviewer rating deviation",
                    "product_rating_deviation": "Product rating deviation",
                    "repeated_word_ratio": "Repeated-word ratio",
                }

                available_evidence = []
                for column, label in evidence_names.items():
                    if column not in case.index:
                        continue
                    value = pd.to_numeric(
                        pd.Series([case[column]]), errors="coerce"
                    ).iloc[0]
                    if pd.isna(value):
                        display_value = "Unavailable"
                    elif column in {
                        "review_text_reuse_ratio",
                        "repeated_word_ratio",
                    }:
                        display_value = f"{value:.1%}"
                    elif column in {
                        "user_rating_deviation",
                        "product_rating_deviation",
                    }:
                        display_value = f"{value:+.2f}"
                    elif column == "days_since_user_previous_review":
                        display_value = f"{value:.1f} days"
                    elif column == "review_text_is_duplicate":
                        display_value = "Yes" if int(value) == 1 else "No"
                    else:
                        display_value = f"{value:,.0f}"
                    available_evidence.append(
                        {"Observed feature": label, "Saved value": display_value}
                    )

                if available_evidence:
                    st.dataframe(
                        pd.DataFrame(available_evidence),
                        width="stretch",
                        hide_index=True,
                    )

                st.caption(
                    "These values are saved predictions and observed features for "
                    "the selected case. They are screening evidence, not proof of "
                    "deception or a causal explanation of the model."
                )

    st.divider()
    st.caption(
        "The summary and calibration metrics below describe saved aggregate "
        "results; they are not recalculated for the selected case."
    )

    st.warning(
        "The Phase 17 score is a relative risk index, not a calibrated "
        "probability that a review is fake."
    )

    if trust:
        a, b, c, d = st.columns(4)
        a.metric("Scored reviews", f"{int(trust.get('rows', 0)):,}")
        b.metric("Mean score", f"{trust.get('score_mean', 0):.3f}")
        c.metric("Median score", f"{trust.get('score_median', 0):.3f}")
        d.metric(
            "Unique row IDs",
            f"{int(trust.get('unique_row_ids', 0)):,}",
        )
        st.write("**Model:**", trust.get("model", "Not specified"))
        st.write("**Interpretation:**", trust.get("interpretation", "Not specified"))
        st.write("**Score range:**",
                 f"{trust.get('score_min', 0):.3f} – "
                 f"{trust.get('score_max', 0):.3f}")
    else:
        st.info("Trust score summary was not found.")

    st.markdown("### Validation calibration audit")
    if calibration:
        a, b, c = st.columns(3)
        a.metric("Actual fake-label rate", pct(calibration.get("actual_fake_rate", 0)))
        b.metric("Mean predicted risk", pct(calibration.get("mean_predicted_risk", 0)))
        c.metric("Brier score", f"{calibration.get('brier_score', 0):.4f}")
        st.write(
            "Weighted absolute calibration gap:",
            f"{calibration.get('weighted_absolute_calibration_gap', 0):.4f}",
        )
        st.write("Validation rows:", f"{int(calibration.get('validation_rows', 0)):,}")
        st.caption(
            "These are validation-set audit values. They are not test-set metrics."
        )
        bins_file = RESULTS / "phase17_validation_calibration_bins.csv"
        if bins_file.exists():
            with st.expander("Calibration bins"):
                st.dataframe(pd.read_csv(bins_file), width="stretch")

elif section == "Explainability":
    st.header("Model Explainability")
    st.write(
        "Permutation importance estimates how much a feature contributes "
        "to predictive performance on the sampled evaluation data. It does "
        "not establish causality."
    )

    report = EXPLAIN / "phase5_explainability_report.md"
    if report.exists():
        with st.expander("Read explainability report", expanded=True):
            st.markdown(report.read_text(encoding="utf-8"))

    for filename, title in [
        ("phase5_baseline_permutation_importance.png", "Baseline model"),
        ("phase5_semantic_permutation_importance.png", "Semantic model"),
    ]:
        path = EXPLAIN / filename
        if path.exists():
            st.markdown(f"### {title}")
            st.image(str(path), width="stretch")

    importance_path = EXPLAIN / "phase5_all_models_permutation_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        with st.expander("Permutation-importance data"):
            st.dataframe(importance, width="stretch")
            st.download_button(
                "Download importance CSV",
                data=importance.to_csv(index=False).encode("utf-8"),
                file_name="phase5_all_models_permutation_importance.csv",
                mime="text/csv",
            )

elif section == "Project Notes":
    st.header("Project Scope and Limitations")
    st.markdown("""
    - The dashboard reads previously saved experiment results.
    - Model evaluation metrics are reported on the held-out YelpZip test split.
    - The dataset's fake/genuine labels are reference labels, not independent
      confirmation of fraudulent behavior.
    - Accuracy should not be interpreted alone because the labels are imbalanced.
    - The Phase 17 trust score is a relative risk index, not a calibrated probability.
    - Permutation importance indicates predictive reliance, not causality.
    - The current dashboard does not score new raw reviews. A verified feature
      generation pipeline is required before adding live scoring.
    """)
    st.markdown("### Source files")
    st.write(
        "Model metrics: `results/phase19_overall_test_metrics.csv`"
    )
    st.write(
        "Trust summary: `results/phase17_trust_intelligence_scores_summary.json`"
    )
    st.write(
        "Calibration audit: `results/phase17_validation_calibration_summary.json`"
    )
    st.write(
        "Explainability: `explainability/phase5_explainability_report.md`"
    )

st.divider()
st.caption(
    "Project Sentinel · Research prototype · Interpret model outputs with care"
)
