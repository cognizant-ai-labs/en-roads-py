"""
Evaluates candidates in order for them to be sorted.
"""
import numpy as np
import pandas as pd
from presp.evaluator import Evaluator
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from constraint.prescriptor import ConstraintPrescriptor
from evolution.data import ContextDataset, SSPDataset
from evolution.outcomes.outcome_manager import OutcomeManager
from enroadspy import load_input_specs
from enroadspy.enroads_runner import EnroadsRunner


class ConstraintEvaluator(Evaluator):
    """
    Evaluates candidates by generating the actions and running the enroads model on them.
    Generates and stores context data based on config using ContextDataset.
    The main workflow is in the evaluate_candidate method:
        1. prescribe_actions
        2. run_enroads
        3. compute_metrics
    """
    def __init__(self,
                 context: list[int],
                 actions: list[int],
                 outcomes: dict[str, bool],
                 n_jobs: int = 1,
                 batch_size: int = 64,
                 device: str = "cpu"):
        outcome_names = list(outcomes.keys())
        super().__init__(outcomes=outcome_names, n_jobs=n_jobs)
        self.actions = list(actions)
        self.outcome_manager = OutcomeManager(outcome_names)
        self.minimize_dict = dict(outcomes)

        # Precise float is required to load the enroads inputs properly
        self.input_specs = load_input_specs()

        self.context = context

        context_data = []
        for i in range(0, 10):
            context_data.append([i / 10, i / 10 + 0.1])
        context_tensor = torch.tensor(context_data, dtype=torch.float32)
        self.context_dataset = TensorDataset(context_tensor)

        self.batch_size = batch_size
        self.device = device

        self.enroads_runner = EnroadsRunner()

    def update_predictor(self, elites):
        pass

    def validate_outcomes(self, outcomes_df: pd.DataFrame):
        """
        Ensures our outcome columns don't have NaNs or infs
        """
        outcome_keys = [key for key in self.outcomes if key in outcomes_df.columns]
        subset = outcomes_df[outcome_keys]
        assert not subset.isna().any().any(), "Outcomes contain NaNs."
        assert not np.isinf(subset.to_numpy()).any(), "Outcomes contain infs."
        return True

    def prescribe_actions(self, candidate: ConstraintPrescriptor, context_dataset: ContextDataset = None) -> list[dict]:
        """
        Takes a candidate, batches contexts, and prescribes actions for each one.
        """
        context_actions_dicts = []
        if context_dataset is None:
            context_dataset = self.context_dataset
        dataloader = DataLoader(context_dataset, batch_size=self.batch_size, shuffle=False)
        # Iterate over batches of contexts
        for batch_context in dataloader:
            actions_dicts = candidate.forward(batch_context[0].to(self.device))
            context_actions_dicts.extend(actions_dicts)

        return context_actions_dicts

    def run_enroads(self, context_actions_dicts: list[dict]) -> list[pd.DataFrame]:
        """
        Runs enroads on context_actions dicts and returns the time series outcomes dfs.
        """
        outcomes_dfs = []
        for context_actions_dict in context_actions_dicts:
            outcomes_df = self.enroads_runner.evaluate_actions(context_actions_dict)
            self.validate_outcomes(outcomes_df)
            outcomes_dfs.append(outcomes_df)
        return outcomes_dfs

    def compute_metrics(self, context_actions_dicts: list[dict], outcomes_dfs: list[pd.DataFrame]) -> np.ndarray:
        """
        Computes the metrics used in evolution with our outcome manager.
        """
        cand_results = []
        for context_actions_dict, outcomes_df in zip(context_actions_dicts, outcomes_dfs):
            results_dict = self.outcome_manager.process_outcomes(context_actions_dict, outcomes_df)
            cand_results.append(results_dict)

        metrics = []
        for key in cand_results[0]:
            metric_results = [result[key] for result in cand_results]
            avg_result = np.mean(metric_results)
            # We always minimize so if we are maximizing we need to negate the result
            if not self.minimize_dict[key]:
                avg_result *= -1
            metrics.append(avg_result)

        return np.array(metrics)

    def evaluate_candidate(self, candidate: ConstraintPrescriptor) -> tuple[np.ndarray, float]:
        """
        Evaluates a single candidate by running all the context through it and receiving all the batches of actions.
        Then evaluates all the actions to get outcomes and computes the average metrics.
        """
        context_actions_dicts = self.prescribe_actions(candidate)
        outcomes_dfs = self.run_enroads(context_actions_dicts)
        metrics = self.compute_metrics(context_actions_dicts, outcomes_dfs)

        return metrics, 0
