import pandas as pd
import plotly.express as px

def plot_outcome_over_time(outcome, cand_idxs, context_idx, all_outcomes_df, baseline_df):
    context_outcomes_df = all_outcomes_df[all_outcomes_df["context_idx"] == context_idx]
    outcomes_df = context_outcomes_df[context_outcomes_df["cand_id"].isin(cand_idxs)]
    fig = px.line(outcomes_df, x="year", y=outcome, color="cand_id", title=f"{outcome} over time")
    return fig