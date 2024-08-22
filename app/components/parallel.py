"""
File containing component in charge of visualizing the candidates' metrics.
"""
from dash import html, dcc, Input
import matplotlib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class ParallelComponent():
    """
    Component displaying the parallel coordinates plot of the prescriptor metrics. Also has a dropdown to select
    candidates to display so the user can compare them.
    """
    def __init__(self, metrics_df: np.ndarray, cand_idxs: list[str], outcomes: dict[str, bool]):
        matplotlib.use('agg')
        self.metrics_df = metrics_df
        self.cand_idxs = cand_idxs + ["baseline", "other"]
        self.outcomes = outcomes
    
    def plot_parallel_coordinates_line(self):
        """
        Plots a parallel coordinates plot of the prescriptor metrics.
        Starts by plotting "other" if selected so that it's the bottom of the z axis.
        Then plots selected candidates in color.
        Finally plots the baseline on top if selected.
        """
        fig = go.Figure()

        normalized_df = self.metrics_df[self.outcomes.keys()]
        normalized_df = (normalized_df - normalized_df.mean()) / (normalized_df.std() + 1e-10)
        normalized_df["cand_id"] = self.metrics_df["cand_id"]

        outcomes_list = list(self.outcomes.keys())
        showlegend=True
        # If "other" is in the cand_idxs, plot all other candidates in lightgray
        if "other" in self.cand_idxs:
            for cand_idx in normalized_df["cand_id"].unique():
                if cand_idx not in self.cand_idxs and cand_idx != "baseline":
                    cand_metrics = normalized_df[normalized_df["cand_id"] == cand_idx]
                    fig.add_trace(go.Scatter(
                        x=outcomes_list,
                        y=cand_metrics.values[0],
                        mode='lines',
                        legendgroup="other",
                        name="other",
                        line=dict(color="lightgray"),
                        showlegend=showlegend
                    ))
                    showlegend=False

        # Plot selected candidates besides baseline so it can be on top
        for cand_idx in self.cand_idxs:
            if cand_idx != "baseline" and cand_idx != "other":
                cand_metrics = normalized_df[normalized_df["cand_id"] == cand_idx]
                fig.add_trace(go.Scatter(
                    x=outcomes_list,
                    y=cand_metrics.values[0],
                    mode='lines',
                    name=str(cand_idx),
                    line=dict(color=px.colors.qualitative.Plotly[self.cand_idxs.index(cand_idx)])
                ))

        # Plot baseline if selected
        if "baseline" in self.cand_idxs:
            baseline_metrics = normalized_df[normalized_df["cand_id"] == "baseline"]
            fig.add_trace(go.Scatter(
                x=outcomes_list,
                y=baseline_metrics.values[0],
                mode='lines',
                name="baseline",
                line=dict(color="black")
            ))

        for i in range(len(outcomes_list)):
            fig.add_vline(x=i, line_color="black")

        fig.update_layout(
            yaxis_range=[normalized_df[outcomes_list].min().min(), normalized_df[outcomes_list].max().max()],
            title={
                'text': "Prescriptor Metrics Averaged Across All SSPs",
                'x': 0.5,  # Center the title
                'xanchor': 'center',  # Anchor it at the center
                'yanchor': 'top'  # Optionally keep it anchored to the top
            }
        )

        return dcc.Graph(figure=fig, id="parallel-coordinates")


    def create_parallel_div(self):
        """
        Creates div showing parallel coordinates plot and dropdown to select candidate(s).
        """
        div = html.Div(
            className="contentBox",
            children=[
                html.H2("Select a Prescriptor to Optimize With", style={"textAlign": "center", "color": "#000048"}),
                html.Div(
                    children=[
                        html.Div(
                            children=[self.plot_parallel_coordinates_line()]
                        ),
                        html.P("Selected Prescriptors", className="centered"),
                        html.Div(
                            style={"width": "50%", "margin-left": "25%", "margin-right": "25%"},
                            children=[
                                dcc.Dropdown(self.cand_idxs,
                                            self.cand_idxs,
                                            id="cand-select-dropdown",
                                            multi=True,
                                            placeholder="Select Candidate(s)")
                            ]
                        )
                    ]
                )
            ]
        )

        return div
