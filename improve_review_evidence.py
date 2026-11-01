from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

old = '''            st.markdown("### Review-specific evidence summary")
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
            )'''

new = '''            st.markdown("### Evidence & interpretation")
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
            )'''

if old not in source:
    raise SystemExit(
        "Could not locate the exact evidence summary block. "
        "No dashboard changes were written."
    )

source = source.replace(old, new, 1)
path.write_text(source, encoding="utf-8")
print("Review evidence cards and analyst takeaway added.")
