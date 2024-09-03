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


class EvolutionHandler():
    """
    Handles evolution results and running of prescriptors for the app.
    """
    def __init__(self):
        save_path = "results/pymoo/context-updated"
        with open(save_path + "/config.json", 'r', encoding="utf-8") as f:
            config = json.load(f)

        self.actions = config["actions"]
        self.outcomes = config["outcomes"]
        # TODO: Make this not hard-coded
        self.model_params = {"in_size": 4, "hidden_size": 16, "out_size": len(self.actions)}

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
            context_tensor = torch.tensor(context_scaled, dtype=torch.float32, device="mps")
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

    def context_baseline_outcomes(self, context_dict: dict[str, float]):
        """
        Takes a context dict and returns the outcomes when no actions are performed.
        """
        return self.runner.evaluate_actions(context_dict)
