from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

replacements = [
(
'''            eligible["selector_label"] = eligible.apply(
                lambda r: f"User {r['user_id']} · {int(r['review_count'])} reviews",
                axis=1,
            )
            selected_label = st.selectbox(
                "Select reviewer",
                eligible["selector_label"].tolist(),
            )
            selected_user = selected_label.split(" · ")[0].replace("User ", "")''',
'''            eligible = eligible.reset_index(drop=True)
            eligible["selector_label"] = eligible.apply(
                lambda r: (
                    f"Reviewer {eligible.index[eligible['user_id'].eq(r['user_id'])[0]] + 1:03d}"
                    f" · {int(r['review_count']):,} reviews"
                ),
                axis=1,
            )
            st.caption(
                "Reviewer names are anonymized in this selector. Review counts "
                "describe activity and are not evidence of deceptive behavior."
            )
            selected_label = st.selectbox(
                "Select reviewer",
                eligible["selector_label"].tolist(),
            )
            selected_index = eligible.index[
                eligible["selector_label"].eq(selected_label)
            ][0]
            selected_user = str(eligible.loc[selected_index, "user_id"])'''
),
(
'''            st.markdown("### Review activity timeline")
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
                )''',
'''            st.markdown("### Review activity timeline")
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
                st.area_chart(monthly)
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
                )'''
),
(
'''            st.markdown("### Rating distribution")
            if "rating" in history.columns:
                ratings = history["rating"].value_counts().sort_index()
                st.bar_chart(ratings.rename_axis("Rating").rename("Reviews"))''',
'''            st.markdown("### Rating distribution")
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
                    )'''
),
(
'''                    show_cols = [
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
                    )''',
'''                    st.caption(
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
                    )'''
)
]

for old, new in replacements:
    if old not in source:
        raise SystemExit(
            "A target section could not be matched. No changes were written."
        )
    source = source.replace(old, new, 1)

path.write_text(source, encoding="utf-8")
print("Reviewer interface improvements applied.")
