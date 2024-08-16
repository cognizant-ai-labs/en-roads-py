import json

import dill
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np

from moo.problems.nn_problem import NNProblem
from enroads_runner import EnroadsRunner
from evolution.outcomes.outcome_manager import OutcomeManager
from generate_url import actions_to_url
from app.components.parallel import plot_parallel_coordinates
from app.components.outcome import plot_outcome_over_time

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

save_path = "results/pymoo/context"
with open(save_path + "/config.json", 'r') as f:
    config = json.load(f)

actions = config["actions"]
outcomes = config["outcomes"]
with open(save_path + "/results", 'rb') as f:
    res = dill.load(f)
    print("Loaded Checkpoint:", res)

X = res.X
X = X[:10,:]
F = res.F

def evenly_sample(lst, m):
    middle = lst[1:-1]
    step = len(middle) / (m-2)
    sample = [middle[int(i * step)] for i in range(m-2)]
    sample = [lst[0]] + sample + [lst[-1]]
    return sample

sort_col_idx = 0
# sample_idxs = evenly_sample(np.argsort(F[:,sort_col_idx]), 10)
sample_idxs = list(range(10))
sample_idxs.append("baseline")

context_df = pd.read_csv("experiments/scenarios/gdp_context.csv")
context_df = context_df.drop(columns=["F", "scenario"])
problem = NNProblem(context_df,
                    {"in_size": len(context_df.columns), "hidden_size": 16, "out_size": len(actions)},
                    actions,
                    outcomes)

runner = EnroadsRunner("app/temp")
baseline_df = runner.evaluate_actions({})
outcome_manager = OutcomeManager(list(outcomes.keys()))
baseline_metrics = outcome_manager.process_outcomes({}, baseline_df)

all_outcomes_dfs = []
all_metrics = []
all_context_actions_dicts = []
for cand_idx in range(X.shape[0]):
    context_actions_dicts = problem.params_to_context_actions_dicts(X[cand_idx])
    all_context_actions_dicts.append(context_actions_dicts)
    cand_outcomes_dfs = problem.run_enroads(context_actions_dicts)
    all_outcomes_dfs.append(cand_outcomes_dfs)
    for i, (context_actions_dict, outcome_df) in enumerate(zip(context_actions_dicts, cand_outcomes_dfs)):
        metrics = outcome_manager.process_outcomes(context_actions_dict, outcome_df)
        metrics["cand_id"] = cand_idx
        metrics["context_idx"] = i
        all_metrics.append(metrics)
        outcome_df["cand_id"] = cand_idx
        outcome_df["context_idx"] = i
        outcome_df["year"] = list(range(1990, 2101))

all_metrics_df = pd.DataFrame(all_metrics)
context_flattened_dfs = [pd.concat(cand_outcomes_dfs, axis=0, ignore_index=True) for cand_outcomes_dfs in all_outcomes_dfs]
all_outcomes_df = pd.concat(context_flattened_dfs, axis=0, ignore_index=True)

# Attach baseline to all_outcomes_df
baseline_df["cand_id"] = "baseline"
baseline_df["year"] = list(range(1990, 2101))
for context_idx in range(len(all_outcomes_dfs[0])):
    baseline_df["context_idx"] = context_idx
    all_outcomes_df = pd.concat([all_outcomes_df, baseline_df], axis=0, ignore_index=True)

baseline_metrics["cand_id"] = "baseline"
# Attach baseline to all_metrics_df
for context_idx in range(len(all_outcomes_dfs[0])):
    baseline_metrics["context_idx"] = context_idx
    all_metrics_df = pd.concat([all_metrics_df, pd.DataFrame([baseline_metrics])], axis=0, ignore_index=True)


context_idx = 0

plot_outcomes = ["Temperature change from 1850", "Adjusted cost of energy per GJ", "Government net revenue from adjustments",  "Total Primary Energy Demand"]

def create_context_scatter(context_chart_df):
    fig = px.scatter(context_chart_df,
                     x="population",
                     y="gdp",
                     color="scenario",
                     labels={"population": "Population 2100 (B)", "gdp": "GDPPP 2100 (T)"},
                     hover_data={"description": True, "population": False, "scenario": False, "gdp": False},
                     title="Select a Context Scenario")
    return fig

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Climate Change Decision Making Page"),
    
    dcc.Graph(figure=create_context_scatter(context_chart_df), id="context-scatter"),
    
    html.Br(),
    
    html.Div(id="parallel-coordinates-container", children=[
        dcc.Graph(id="parallel-coordinates")
    ]),
    html.Div(children=[
        dcc.Graph(id=f"outcome-{i}") for i in range(len(plot_outcomes))
    ]),
    dcc.Dropdown(sample_idxs, sample_idxs[0], id="cand-dropdown"),

    html.A("View Candidate", id="cand-link", href="#")
])

@app.callback(
    Output("parallel-coordinates-container", "children"),
    Input("context-scatter", "clickData")
)
def click_context(click_data):
    """
    Selects context when point on map is clicked.
    :param click_data: Input data from click action.
    :return: The new longitude and latitude to put into the dropdowns.
    """
    if click_data:
        scenario = int(click_data["points"][0]["customdata"][1][-1]) - 1
        
    else:
        scenario = 0

    fig = plot_parallel_coordinates(all_metrics_df, scenario, sample_idxs, outcomes)
    
    return dcc.Graph(figure=fig, id="parallel-coordinates")

@app.callback(
    [Output(f"outcome-{i}", "figure") for i in range(len(plot_outcomes))],
    Input("context-scatter", "clickData"),
)
def update_outcomes_plots(click_data):
    if click_data:
        scenario = int(click_data["points"][0]["customdata"][1][-1]) - 1
    else:
        scenario = 0
    
    figs = []
    for outcome in plot_outcomes:
        fig = plot_outcome_over_time(outcome, sample_idxs, scenario, all_outcomes_df)
        figs.append(fig)
    return figs

@app.callback(
    Output("cand-link", "href"),
    Input("cand-dropdown", "value"),
    Input("context-scatter", "clickData")
)
def update_cand_link(cand_idx, click_data):
    if click_data:
        scenario = int(click_data["points"][0]["customdata"][1][-1]) - 1
    else:
        scenario = 0
    url = actions_to_url(all_context_actions_dicts[cand_idx][scenario])
    return url

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)