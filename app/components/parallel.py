import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from io import BytesIO
import base64

def plot_parallel_coordinates(metrics_df, context_idx, cand_idxs, outcomes, baseline_metrics):
    baseline_dict = {outcome: metric for outcome, metric in baseline_metrics.items()}
    baseline_dict["cand_id"] = 999
    baseline_df = pd.DataFrame([baseline_dict])
    
    coords_df = metrics_df[metrics_df["context_idx"] == context_idx]
    coords_df = coords_df[coords_df["cand_id"].isin(cand_idxs)]
    coords_df = pd.concat([coords_df, baseline_df], ignore_index=True, copy=True)
    coords_df = coords_df.sort_values("cand_id")
    coords_df["color"] = list(range(len(coords_df)))
    color_map = [c for c in px.colors.qualitative.Plotly]
    if len(coords_df) < len(color_map):
        color_map[len(coords_df)-1] = "black"
    else:
        color_map.append("black")
    fig = px.parallel_coordinates(
        coords_df,
        dimensions=list(outcomes.keys()),
        title=f"Metrics for SSP {context_idx+1}",
        color="color",
        color_continuous_scale=color_map
    )
    fig.update_coloraxes(showscale=False)

    return fig

# def matplotlib_parallel_coordinates(metrics_df, context_idx, cand_idxs, outcomes, baseline_metrics):
#     baseline_dict = {outcome: metric for outcome, metric in baseline_metrics.items()}
#     baseline_dict["cand_id"] = "Baseline"
#     baseline_df = pd.DataFrame([baseline_dict])
    
#     coords_df = metrics_df[metrics_df["context_idx"] == context_idx]
#     coords_df = pd.concat([coords_df, baseline_df], ignore_index=True, copy=True)
#     coords_df['cand_idx'] = coords_df['cand_idx'].apply(lambda x: x if x in cand_idxs + ["Baseline"] else 'other')
#     normalized_df = (coords_df - coords_df.mean()) / coords_df.std()

#     pd.plotting.parallel_coordinates(normalized_df[normalized_df["cand_id"] == "other"], "cand_id", color=["lightgray"])

#     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     # Color baseline black
#     condition = (normalized_df["cand_id"] != "other") & (normalized_df["cand_id"] != "Baseline")
#     pd.plotting.parallel_coordinates(normalized_df[condition], "cand_id", color=colors)

#     pd.plotting.parallel_coordinates(normalized_df[normalized_df["cand_id"] == "Baseline"], "cand_id", color="black")
#     plt.legend(bbox_to_anchor=(1, 1))
#     plt.xticks(rotation=90)
#     plt.ylabel("Normalized Value")
#     plt.title(f"Parallel Coordinates of SSP {context_idx + 1}")

#     buf = BytesIO()
#     plt.savefig(buf, format="png")
#     fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
#     fig_parcoord_plt = f"data:image/png;base64,{fig_data}"

#     fig_plotly = px.