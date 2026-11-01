from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

replacements = [
(
'''            st.markdown("### Review volume over time")
            st.area_chart(trend["reviews"])
            st.markdown("### Active reviewers over time")
            st.line_chart(trend["unique_reviewers"])
            st.markdown("### Average rating over time")
            st.line_chart(trend["average_rating"])''',
'''            st.markdown("### Review volume over time")
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
                )'''
),
(
'''                st.line_chart(risk_trend["mean_estimated_risk"])
                st.bar_chart(risk_trend["model_flagged"])''',
'''                st.markdown("#### Average estimated risk by period")
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
                )'''
)
]

for old, new in replacements:
    if old not in source:
        raise SystemExit(
            "Expected temporal code block was not found. "
            "No dashboard changes were written."
        )
    source = source.replace(old, new, 1)

path.write_text(source, encoding="utf-8")
print("Temporal chart explanations and derived insights added.")
