import pandas as pd
import plotly.express as px

def plot_outcome_over_time(outcome, cand_idxs, context_idx, all_outcomes_df):
    min_val = all_outcomes_df[outcome].min()
    max_val = all_outcomes_df[outcome].max()
    context_outcomes_df = all_outcomes_df[all_outcomes_df["context_idx"] == context_idx]
    outcomes_df = context_outcomes_df[context_outcomes_df["cand_id"].isin(cand_idxs)]
    color_map = [c for c in px.colors.qualitative.Plotly]
    if len(cand_idxs) < len(color_map):
        color_map[len(cand_idxs)-1] = "black"
    else:
        color_map.append("black")
    fig = px.line(outcomes_df,
                  x="year",
                  y=outcome,
                  color="cand_id",
                  range_y=[min_val, max_val],
                  color_discrete_sequence=color_map)
    fig.update_layout(title={
        "text": f"{outcome} Over Time",
        "x": 0.5,
        "xanchor": "center"
    })
    return fig