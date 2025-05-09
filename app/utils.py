"""
Utilities for the demo app.
"""
from pathlib import Path

import pandas as pd
from presp.prescriptor import NNPrescriptorFactory
import yaml

from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.data import ContextDataset
from evolution.evaluation.evaluator import EnROADSEvaluator
from evolution.utils import process_config


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
    def __init__(self, save_path: str):
        save_path = Path(save_path)
        with open(save_path / "config.yml", 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config = process_config(config)

        self.context = config["context"]
        self.actions = config["actions"]
        # TODO: This is hardcoded for now, not sure whether to make it on the user to modify the config or the app to
        # fix the ordering.
        metrics = ["Temperature change from 1850",
                   "Max cost of energy",
                   "Government net revenue below zero",
                   "Total energy below baseline"]
        self.outcomes = {metric: config["outcomes"][metric] for metric in metrics}
        self.model_params = config["model_params"]

        self.factory = NNPrescriptorFactory(EnROADSPrescriptor,
                                            self.model_params,
                                            device="cpu",
                                            actions=self.actions)

        self.evaluator = EnROADSEvaluator(self.context,
                                          self.actions,
                                          self.outcomes,
                                          n_jobs=1,
                                          batch_size=1,
                                          device="cpu",
                                          decomplexify=False)

        self.population = self.factory.load_population(save_path / "population")
        # Get the order in which the candidates should be prescribed with
        results_df = pd.read_csv(save_path / "results.csv")
        results_df = results_df[(results_df["gen"] == results_df["gen"].max()) & (results_df["rank"] == 1)]
        self.cand_ids = results_df["cand_id"].tolist()

    def prescribe_all(self, context_dict: dict[int, float]) -> list[dict[int, float]]:
        """
        Takes a dict containing a single context and prescribes actions for it using all the candidates.
        Returns a context_actions dict for each candidate.
        """
        context_df = pd.DataFrame([context_dict])
        # TODO: This is a bit of a hack to pull the scaler out
        scaler = self.evaluator.context_dataset.scaler
        context_ds = ContextDataset(context_df, scaler=scaler)

        context_actions_dicts = []
        for cand_id in self.cand_ids:
            context_actions_dicts.append(self.evaluator.prescribe_actions(self.population[cand_id], context_ds)[0])

        return context_actions_dicts

    def context_actions_to_outcomes(self, context_actions_dicts: list[dict[int, float]]) -> list[pd.DataFrame]:
        """
        Takes a context dict and prescribes actions for it. Then runs enroads on those actions and returns the outcomes.
        """
        outcomes_dfs = self.evaluator.run_enroads(context_actions_dicts)
        return outcomes_dfs

    def outcomes_to_metrics(self,
                            context_actions_dicts: list[dict[int, float]],
                            outcomes_dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Takes parallel lists of context_actions_dicts and outcomes_dfs and processes them into a metrics dict.
        All of these metrics dicts are then concatenated into a single DataFrame.
        TODO: We hard-code some metrics to be more app-friendly
        """
        cand_results = []
        for context_actions_dict, outcomes_df in zip(context_actions_dicts, outcomes_dfs):
            results_dict = self.evaluator.outcome_manager.process_outcomes(context_actions_dict, outcomes_df)
            cand_results.append(results_dict)

        metrics_df = pd.DataFrame(cand_results)
        return metrics_df

    def context_baseline_outcomes(self, context_dict: dict[int, float]) -> pd.DataFrame:
        """
        Takes a context dict and returns the outcomes when no actions are performed.
        """
        return self.evaluator.run_enroads([context_dict])[0]
