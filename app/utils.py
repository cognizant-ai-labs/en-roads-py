"""
Utilities for the demo app.
"""
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from enroadspy.enroads_runner import EnroadsRunner
from evolution.candidate import Candidate
from evolution.outcomes.outcome_manager import OutcomeManager


def filter_metrics_json(metrics_json: dict[str, list],
                        metric_ranges: list[tuple[float, float]],
                        normalize=False) -> pd.DataFrame:
    """
    Converts metrics json stored in the metrics store to a DataFrame then filters it based on metric ranges from
    sliders.
    """
    metrics_df = pd.DataFrame(metrics_json)
    mu = metrics_df.mean()
    sigma = metrics_df.std()
    metric_names = metrics_df.columns
    metric_name_and_range = zip(metric_names, metric_ranges)
    for metric_name, metric_range in metric_name_and_range:
        # Never filter out the baseline
        condition = (metrics_df[metric_name].between(*metric_range)) | (metrics_df.index == metrics_df.index[-1])
        metrics_df = metrics_df[condition]
    if normalize:
        metrics_df = (metrics_df - mu) / (sigma + 1e-10)
    return metrics_df


class EvolutionHandler():
    """
    Handles evolution results and running of prescriptors for the app.
    """
    def __init__(self):
        save_path = "app/results"
        with open(save_path + "/config.json", 'r', encoding="utf-8") as f:
            config = json.load(f)

        self.actions = config["actions"]
        self.outcomes = config["outcomes"]
        # TODO: Make this not hard-coded
        self.model_params = {"in_size": 4, "hidden_size": 16, "out_size": len(self.actions)}
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        self.X = np.load(save_path + "/X.npy")
        self.F = np.load(save_path + "/F.npy")

        context_df = pd.read_csv("experiments/scenarios/gdp_context.csv")
        self.context_df = context_df.drop(columns=["F", "scenario"])
        self.scaler = StandardScaler()
        self.scaler.fit(self.context_df.to_numpy())

        self.runner = EnroadsRunner()
        self.outcome_manager = OutcomeManager(list(self.outcomes.keys()))

    def load_initial_metrics_df(self):
        """
        Takes the F results matrix and converts it into a DataFrame the way pandas parcoords wants it. We also attach
        the average of the baseline over all the contexts to this DataFrame.
        """
        # Convert F to DataFrame
        metrics_df = pd.DataFrame(self.F, columns=list(self.outcomes.keys()))
        for outcome, ascending in self.outcomes.items():
            if not ascending:
                metrics_df[outcome] *= -1
        metrics_df["cand_id"] = range(len(self.F))

        # Run En-ROADS on baseline over all contexts
        baseline_metrics_avg = {outcome: 0 for outcome in self.outcomes}
        for _, row in self.context_df.iterrows():
            context_dict = row.to_dict()
            baseline_outcomes = self.runner.evaluate_actions(context_dict)
            baseline_metrics = self.outcome_manager.process_outcomes(context_dict, baseline_outcomes)
            for outcome, val in baseline_metrics.items():
                baseline_metrics_avg[outcome] += val

        # Finish preprocessing baseline metrics
        for outcome in self.outcomes:
            baseline_metrics_avg[outcome] /= len(self.context_df)
        baseline_metrics_avg["cand_id"] = "baseline"

        # Attach baseline to metrics_df
        metrics_df = pd.concat([metrics_df, pd.DataFrame([baseline_metrics_avg])], axis=0, ignore_index=True)

        # TODO: Eventually don't hard-code this. Flip the net revenue below 0 to be something we minimize
        if "Government net revenue below zero" in metrics_df.columns:
            metrics_df["Government net revenue below zero"] *= -1
        if "Total energy below baseline" in metrics_df.columns:
            metrics_df["Total energy below baseline"] *= -1

        return metrics_df

    def prescribe_all(self, context_dict: dict[str, float]):
        """
        Takes a dict containing a single context and prescribes actions for it using all the candidates.
        Returns a context_actions dict for each candidate.
        """
        context_actions_dicts = []
        for x in self.X:
            candidate = Candidate.from_pymoo_params(x, self.model_params, self.actions, self.outcomes)
            # Process context_dict into tensor
            context_list = [context_dict[context] for context in self.context_df.columns]
            context_scaled = self.scaler.transform([context_list])
            context_tensor = torch.tensor(context_scaled, dtype=torch.float32, device=self.device)
            actions_dict = candidate.prescribe(context_tensor)[0]
            actions_dict.update(context_dict)
            context_actions_dicts.append(actions_dict)

        return context_actions_dicts

    def context_actions_to_outcomes(self, context_actions_dicts: list[dict[str, float]]):
        """
        Takes a context dict and prescribes actions for it. Then runs enroads on those actions and returns the outcomes.
        """
        outcomes_dfs = []
        for context_actions_dict in context_actions_dicts:
            outcomes_df = self.runner.evaluate_actions(context_actions_dict)
            outcomes_dfs.append(outcomes_df)

        return outcomes_dfs

    def outcomes_to_metrics(self,
                            context_actions_dicts: list[dict[str, float]],
                            outcomes_dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Takes parallel lists of context_actions_dicts and outcomes_dfs and processes them into a metrics dict.
        All of these metrics dicts are then concatenated into a single DataFrame.
        """
        metrics_dicts = []
        for context_actions_dict, outcomes_df in zip(context_actions_dicts, outcomes_dfs):
            metrics = self.outcome_manager.process_outcomes(context_actions_dict, outcomes_df)
            metrics_dicts.append(metrics)

        metrics_df = pd.DataFrame(metrics_dicts)
        return metrics_df

    def context_baseline_outcomes(self, context_dict: dict[str, float]):
        """
        Takes a context dict and returns the outcomes when no actions are performed.
        """
        return self.runner.evaluate_actions(context_dict)
