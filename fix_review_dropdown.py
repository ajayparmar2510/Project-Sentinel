from pathlib import Path

p = Path("dashboard.py")
s = p.read_text(encoding="utf-8")

start = s.find('            default_review_id = int(risk_order.iloc[0]["row_id"])')
end = s.find('            score = float(review["semantic_probability"])', start)

if start == -1 or end == -1:
    raise SystemExit("Could not find the selection block. Dashboard unchanged.")

replacement = '''            choices = risk_order.head(100).copy()
            choices["row_id"] = choices["row_id"].astype(int)

            choice = st.selectbox(
                "Choose a high-risk test review",
                options=choices["row_id"].tolist(),
                format_func=lambda rid: (
                    f"Review #{rid} — "
                    f"{choices.loc[choices['row_id'].eq(rid), 'semantic_probability'].iloc[0]:.1%} estimated risk"
                ),
            )

            review = review_data.loc[
                review_data["row_id"].eq(int(choice))
            ].iloc[0]

'''

p.write_text(s[:start] + replacement + s[end:], encoding="utf-8")
print("Updated selector to use valid high-risk review IDs.")
