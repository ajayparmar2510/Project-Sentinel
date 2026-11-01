from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

old = '''            selected_id = st.selectbox(
                "Search all valid test review IDs",
                options=valid_ids,
                index=valid_ids.index(int(risk_order.iloc[0]["row_id"])),
                format_func=lambda rid: f"Review #{rid}",
            )'''

new = '''            st.markdown("### Find a review")
            st.caption(
                "Choose a case by its estimated model risk. Internal dataset IDs "
                "are kept out of the visible selector."
            )

            def case_label(rid):
                row = review_data.loc[review_data["row_id"].eq(int(rid))].iloc[0]
                risk = float(row["semantic_probability"])
                if risk >= 0.60:
                    band = "Elevated estimated risk"
                elif risk >= threshold:
                    band = "Flagged for review"
                else:
                    band = "Lower estimated risk"
                return f"Case {valid_ids.index(int(rid)) + 1:03d}  ·  {band}  ·  {risk:.0%}"

            selected_id = st.selectbox(
                "Review case",
                options=valid_ids,
                index=valid_ids.index(int(risk_order.iloc[0]["row_id"])),
                format_func=case_label,
                help="Risk is a model estimate used to prioritize human review, not proof of deception.",
            )'''

if old not in source:
    raise SystemExit(
        "Could not find the exact existing selector. No changes were made."
    )

source = source.replace(old, new, 1)
path.write_text(source, encoding="utf-8")
print("Review selector updated. Internal row IDs remain unchanged.")
