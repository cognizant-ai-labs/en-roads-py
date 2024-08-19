import plotly.express as px
import plotly.graph_objects as go


def plot_parallel_coordinates(metrics_df, context_idx, cand_idxs, outcomes):
    """
    Takes metrics df of all metrics for all candidates and contexts, and plots parallel coordinates for given context
    for the candidate indices given over the outcomes given.
    """
    coords_df = metrics_df[metrics_df["context_idx"] == context_idx]
    coords_df = coords_df[coords_df["cand_id"].isin(cand_idxs)]
    coords_df["cand_id"] = coords_df["cand_id"].astype(str)
    # coords_df = coords_df.sort_values("cand_id")
    coords_df["color"] = list(range(len(coords_df)))
    color_map = [c for c in px.colors.qualitative.Plotly]
    if len(coords_df) < len(color_map):
        color_map[len(coords_df)-1] = "black"
    else:
        color_map.append("black")

    fig = go.Figure(data =
        go.Parcoords(
            line=dict(color=coords_df["color"], colorscale=color_map),
            dimensions = list([
                dict(range = [metrics_df[outcome].min(), metrics_df[outcome].max()],
                     label = outcome,
                     values = coords_df[outcome]) for outcome in outcomes.keys()
            ])
        )
    )

    fig.update_layout(title={
        "text": f"Metrics for SSP {context_idx+1}",
        "x": 0.5,
        "xanchor": "center"
    })
    fig.update_coloraxes(showscale=False)

    return fig
