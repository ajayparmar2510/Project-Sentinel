from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

if "import plotly.express as px" not in source:
    source = source.replace(
        "import pandas as pd",
        "import pandas as pd\nimport plotly.express as px",
        1,
    )

old = "        st.bar_chart(overview)"

new = '''        chart_data = overview.rename_axis("Metric").reset_index().melt(
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
        st.plotly_chart(fig, use_container_width=True)'''

if old not in source:
    raise SystemExit(
        "Could not find the expected Overview chart line. "
        "dashboard.py was not modified."
    )

source = source.replace(old, new, 1)
path.write_text(source, encoding="utf-8")
print("Overview chart upgraded to interactive Plotly.")
