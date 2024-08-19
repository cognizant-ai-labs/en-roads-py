"""
Demo app for En-ROADS optimization results.
"""
import json

import dill
import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd

from enroads_runner import EnroadsRunner
from evolution.outcomes.outcome_manager import OutcomeManager
from generate_url import actions_to_url
from moo.problems.nn_problem import NNProblem

from app.components.context import create_context_scatter
from app.components.outcome import plot_outcome_over_time
from app.components.parallel import plot_parallel_coordinates


save_path = "results/pymoo/context-updated"
with open(save_path + "/config.json", 'r', encoding="utf-8") as f:
    config = json.load(f)

actions = config["actions"]
outcomes = config["outcomes"]
with open(save_path + "/results", 'rb') as f:
    res = dill.load(f)
    print("Loaded Checkpoint:", res)

X = res.X
F = res.F

def evenly_sample(lst, m):
    middle = lst[1:-1]
    step = len(middle) / (m-2)
    sample = [middle[int(i * step)] for i in range(m-2)]
    sample = [lst[0]] + sample + [lst[-1]]
    return sample

sort_col_idx = 0
sample_idxs = evenly_sample(np.argsort(F[:,sort_col_idx]), 10)
# sample_idxs = list(range(10))
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
for cand_idx in sample_idxs[:-1]:
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

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Climate Change Decision Making"

@app.callback(
    Output("parallel-coordinates", "figure"),
    Input("context-scatter", "clickData")
)
def click_context(click_data):
    """
    Updates parallel coordinates when a context scatter point is clicked.
    :param click_data: Input data from click action.
    :return: The parallel coordinates plot.
    """
    if click_data:
        scenario = int(click_data["points"][0]["customdata"][1][-1]) - 1

    else:
        scenario = 0

    fig = plot_parallel_coordinates(all_metrics_df, scenario, sample_idxs, outcomes)

    return fig

@app.callback(
    Output("outcome-graph-1", "figure"),
    Input("outcome-dropdown-1", "value"),
    Input("context-scatter", "clickData")
)
def update_outcomes_plot_1(outcome, click_data):
    """
    Updates outcome plot when specific outcome is selected or context scatter point is clicked.
    """
    if click_data:
        scenario = int(click_data["points"][0]["customdata"][1][-1]) - 1
    else:
        scenario = 0

    fig = plot_outcome_over_time(outcome, sample_idxs, scenario, all_outcomes_df)
    return fig

@app.callback(
    Output("outcome-graph-2", "figure"),
    Input("outcome-dropdown-2", "value"),
    Input("context-scatter", "clickData")
)
def update_outcomes_plot_2(outcome, click_data):
    """
    Updates outcome plot when specific outcome is selected or context scatter point is clicked.
    """
    if click_data:
        scenario = int(click_data["points"][0]["customdata"][1][-1]) - 1
    else:
        scenario = 0

    fig = plot_outcome_over_time(outcome, sample_idxs, scenario, all_outcomes_df)
    return fig

@app.callback(
    Output("cand-link", "href"),
    Input("cand-dropdown", "value"),
    Input("context-scatter", "clickData")
)
def update_cand_link(cand, click_data):
    """
    Updates the candidate link when a specific candidate is selected.
    """
    if click_data:
        scenario = int(click_data["points"][0]["customdata"][1][-1]) - 1
    else:
        scenario = 0
    url = actions_to_url(all_context_actions_dicts[sample_idxs.index(cand)][scenario])
    return url

# Layout of the app
app.layout = html.Div([
    html.H1("Climate Change Decision Making Page", style={"textAlign": "center"}),

    dcc.Graph(id="context-scatter", figure=create_context_scatter()),

    dcc.Graph(id="parallel-coordinates"),

    html.Div([
        html.Div([
            dcc.Dropdown(plot_outcomes, plot_outcomes[0], id="outcome-dropdown-1")
        ], style={"width": "50%", "display": "inline-block"}),
        html.Div([
            dcc.Dropdown(plot_outcomes, plot_outcomes[1], id="outcome-dropdown-2")
        ], style={"width": "50%", "display": "inline-block"}),
    ]),
    html.Div([
        html.Div([
            dcc.Graph(id="outcome-graph-1"),
        ], style={"width": "50%", "display": "inline-block"}),
        html.Div([
            dcc.Graph(id="outcome-graph-2"),
        ], style={"width": "50%", "display": "inline-block"}),
    ]),

    dcc.Dropdown(sample_idxs[:-1], sample_idxs[0], id="cand-dropdown"),

    html.A("View Candidate", id="cand-link", href="#", target="_blank", rel="noopener noreferrer")
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)