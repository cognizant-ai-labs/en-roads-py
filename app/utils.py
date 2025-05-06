"""
Utilities for the demo app.
"""
from pathlib import Path

import pandas as pd
from presp.prescriptor import NNPrescriptorFactory
import torch
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
    TODO: Currently we hard-code some metrics to make the app prettier. Later we should just create more app-friendly
    metrics to optimize for.
    """
    def __init__(self):
        save_path = Path("app/results")
        with open(save_path / "config.yml", 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config = process_config(config)

        self.context = config["context"]
        self.actions = config["actions"]
        self.outcomes = config["outcomes"]
        self.model_params = config["model_params"]
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

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

        self.candidates = []
        pareto_df = pd.read_csv(save_path / f"{config['evolution_params']['n_generations']}.csv")
        pareto_df = pareto_df[pareto_df["rank"] == 1]
        for cand_id in pareto_df["cand_id"]:
            self.candidates.append(self.factory.load(save_path / f"{cand_id.split('_')[0]}" / f"{cand_id}"))

    def prescribe_all(self, context_dict: dict[str, float]) -> list[dict[str, float]]:
        """
        Takes a dict containing a single context and prescribes actions for it using all the candidates.
        Returns a context_actions dict for each candidate.
        """
        context_df = pd.DataFrame([context_dict])
        # TODO: This is a bit of a hack to pull the scaler out
        scaler = self.evaluator.context_dataset.scaler
        context_ds = ContextDataset(context_df, scaler=scaler)

        context_actions_dicts = []
        for candidate in self.candidates:
            context_actions_dicts.append(self.evaluator.prescribe_actions(candidate, context_ds)[0])

        return context_actions_dicts

    def context_actions_to_outcomes(self, context_actions_dicts: list[dict[str, float]]) -> list[pd.DataFrame]:
        """
        Takes a context dict and prescribes actions for it. Then runs enroads on those actions and returns the outcomes.
        """
        outcomes_dfs = self.evaluator.run_enroads(context_actions_dicts)
        return outcomes_dfs

    def outcomes_to_metrics(self,
                            context_actions_dicts: list[dict[str, float]],
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

    def context_baseline_outcomes(self, context_dict: dict[str, float]) -> pd.DataFrame:
        """
        Takes a context dict and returns the outcomes when no actions are performed.
        """
        return self.evaluator.run_enroads([context_dict])[0]
