"""
Evaluates candidates in order for them to be sorted.
"""
import numpy as np
import pandas as pd
from presp.evaluator import Evaluator
import torch
from torch.utils.data import DataLoader, Subset

from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.data import SSPDataset
from evolution.outcomes.outcome_manager import OutcomeManager
from enroadspy import load_input_specs
from enroadspy.enroads_runner import EnroadsRunner


class EnROADSEvaluator(Evaluator):
    """
    Evaluates candidates by generating the actions and running the enroads model on them.
    Generates and stores context data based on config using ContextDataset.
    """
    def __init__(self,
                 context: list[str],
                 actions: list[str],
                 outcomes: dict[str, bool],
                 n_jobs: int = 1,
                 batch_size: int = 64,
                 device: str = "cpu",
                 decomplexify: bool = False):
        outcome_names = list(outcomes.keys())
        super().__init__(outcomes=outcome_names, n_jobs=n_jobs)
        self.actions = actions
        self.outcome_manager = OutcomeManager(outcome_names)
        self.minimize_dict = {o: m for o, m in outcomes.items()}

        # Precise float is required to load the enroads inputs properly
        self.input_specs = load_input_specs()

        self.context = context
        # Context Dataset outputs a scaled tensor and nonscaled tensor. The scaled tensor goes into PyTorch and
        # the nonscaled tensor is used to reconstruct the context that goes into enroads.
        if set(context) == {"_global_population_in_2100",
                            "_long_term_gdp_per_capita_rate",
                            "_near_term_gdp_per_capita_rate",
                            "_transition_time_to_reach_long_term_gdp_per_capita_rate"}:
            self.context_dataset = SSPDataset()
        # NOTE: This is a hack. We don't actually pass context into the direct prescriptor so we just grab the first
        # example to force the direct prescriptor to return a single actions dict
        elif len(context) == 0:
            self.context_dataset = Subset(SSPDataset(), range(0, 1))
        else:
            raise ValueError(f"Context {context} not recognized.")

        self.batch_size = batch_size
        self.device = device

        # Get the switches that should always be on when decomplexifying
        self.decomplexify = decomplexify
        if self.decomplexify:
            input_specs = load_input_specs()
            condition = (
                (input_specs["kind"] == "switch") &
                (input_specs["varId"] != "_electric_standard_active") &
                (input_specs["slidersActiveWhenOn"].apply(lambda x: isinstance(x, list) and len(x) > 0))
            )
            always_on_switches = input_specs.loc[condition, "varId"]
            always_on_values = input_specs.loc[condition, "onValue"]
            self.decomplexify_dict = dict(zip(always_on_switches, always_on_values))

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

    def reconstruct_context_dicts(self, batch_context: torch.Tensor) -> list[dict[str, float]]:
        """
        Takes a torch tensor and zips it with the context labels to create a list of dicts.
        """
        context_dicts = []
        for row in batch_context:
            context_dict = dict(zip(self.context, row.tolist()))
            context_dicts.append(context_dict)
        return context_dicts

    def prescribe_actions(self, candidate: EnROADSPrescriptor) -> list[dict]:
        """
        Takes a candidate, batches contexts, and prescribes actions for each one. Then attaches the context to the
        actions to return context_actions dicts.
        """
        context_actions_dicts = []
        dataloader = DataLoader(self.context_dataset, batch_size=self.batch_size, shuffle=False)
        # Iterate over batches of contexts
        for batch_tensor, batch_context in dataloader:
            context_dicts = self.reconstruct_context_dicts(batch_context)
            actions_dicts = candidate.forward(batch_tensor.to(self.device))
            for actions_dict, context_dict in zip(actions_dicts, context_dicts):
                # Add context to actions so we can pass it into the model
                actions_dict.update(context_dict)
                context_actions_dicts.append(actions_dict)

        return context_actions_dicts

    def run_enroads(self, context_actions_dicts: list[dict]) -> list[pd.DataFrame]:
        """
        Runs enroads on context_actions dicts and returns the time series outcomes dfs.
        If decomplexify is active, we activate the switches to decomplexify the model.
        """
        outcomes_dfs = []
        for context_actions_dict in context_actions_dicts:
            # If decomplexify is active, set the decomplexify switches to on
            if self.decomplexify:
                # We don't want to overwrite the context actions dicts so we create a new dict
                context_actions_dict = dict(context_actions_dict)
                context_actions_dict.update(self.decomplexify_dict)
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

    def evaluate_candidate(self, candidate: EnROADSPrescriptor) -> np.ndarray:
        """
        Evaluates a single candidate by running all the context through it and receiving all the batches of actions.
        Then evaluates all the actions to get outcomes and computes the average metrics.
        """
        context_actions_dicts = self.prescribe_actions(candidate)
        outcomes_dfs = self.run_enroads(context_actions_dicts)
        metrics = self.compute_metrics(context_actions_dicts, outcomes_dfs)

        return metrics
