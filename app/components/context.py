import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_context_scatter():
    ar6_df = pd.read_csv("experiments/scenarios/ar6_snapshot_1723566520.csv/ar6_snapshot_1723566520.csv")
    ar6_df = ar6_df.dropna(subset=["Scenario"])
    ar6_df = ar6_df.dropna(axis=1)
    ar6_df = ar6_df[ar6_df["Scenario"].str.contains("Baseline")]
    pop_df = ar6_df[ar6_df["Variable"] == "Population"]
    gdp_df = ar6_df[ar6_df["Variable"] == "GDP|PPP"]

    context_chart_df = pd.DataFrame()
    context_chart_df["scenario"] = ar6_df["Scenario"].unique()
    context_chart_df["scenario"] = context_chart_df["scenario"].str.split("-", expand=True)[0]
    context_chart_df["population"] = pop_df["2100"].values / 1000
    context_chart_df["gdp"] = gdp_df["2100"].values / 1000
    context_chart_df = context_chart_df.sort_values(by="scenario")
    context_chart_df["description"] = ["SSP1: Sustainable Development",
                                "SSP2: Middle of the Road",
                                "SSP3: Regional Rivalry",
                                "SSP4: Inequality",
                                "SSP5: Fossil-Fueled"]

    fig = px.scatter(context_chart_df,
                     x="population",
                     y="gdp",
                     color="scenario",
                     labels={"population": "Population 2100 (B)", "gdp": "GDPPP 2100 (T)"},
                     hover_data={"description": True, "population": False, "scenario": False, "gdp": False})


    fig.update_layout(
        title = {
            "text": "Select a Context Scenario",
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis_title="Population 2100 (B)",
        yaxis_title="GDP 2100 (T$)",
        showlegend=True
    )
    return fig