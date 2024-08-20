import base64
import io

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from dash import html, dcc

class ParallelComponent():

    def __init__(self, metrics_df: np.ndarray, cand_idxs: list[str], outcomes: dict[str, bool]):
        self.metrics_df = metrics_df
        self.cand_idxs = cand_idxs
        self.outcomes = outcomes

    def plot_parallel_coordinates_plt(self):
        normalized_df = self.metrics_df[self.outcomes.keys()]
        normalized_df = (normalized_df - normalized_df.mean()) / (normalized_df.std() + 1e-10)
        normalized_df["cand_id"] = self.metrics_df["cand_id"]

        condition = ~normalized_df["cand_id"].isin(self.cand_idxs) & (normalized_df["cand_id"] != "baseline")
        other_df = normalized_df[condition].copy()
        other_df["cand_id"] = "other"
        pd.plotting.parallel_coordinates(other_df, "cand_id", self.outcomes.keys(), color="lightgray")

        cand_df = normalized_df[normalized_df["cand_id"].isin(self.cand_idxs)]
        colors = [c for c in px.colors.qualitative.Plotly]
        pd.plotting.parallel_coordinates(cand_df, "cand_id", self.outcomes.keys(), color=colors)

        baseline_df = normalized_df[normalized_df["cand_id"] == "baseline"]
        pd.plotting.parallel_coordinates(baseline_df, "cand_id", self.outcomes.keys(), color="black")

        plt.title("Average Metrics for All SSPs")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("utf-8")
        buf.close()
        return f"data:image/png;base64,{data}"
        
    def create_parallel_div(self):
        
        div = html.Div([
            html.Img(id="parallel-coordinates", src=self.plot_parallel_coordinates_plt()),
            dcc.Dropdown(self.cand_idxs, [], id="cand-select-dropdown", multi=True)
        ])

        return div