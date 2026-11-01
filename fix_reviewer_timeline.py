from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

old = '''                st.area_chart(monthly)'''

new = '''                monthly_df = monthly.rename_axis("Month").reset_index()
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
                st.plotly_chart(fig, use_container_width=True)'''

if source.count(old) != 1:
    raise SystemExit(
        f"Expected exactly one reviewer timeline chart; found {source.count(old)}. "
        "No changes were made."
    )

source = source.replace(old, new, 1)
path.write_text(source, encoding="utf-8")
print("Reviewer timeline now uses an explicit date axis.")
