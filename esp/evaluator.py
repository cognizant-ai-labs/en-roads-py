"""
Evaluates candidates in order for them to be sorted.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from presp.evaluator import Evaluator
import torch
from torch.utils.data import Dataset, DataLoader

from esp.prescriptor import EnROADSPrescriptor
from esp.outcome_manager import OutcomeManager
from enroadspy.enroads_runner import EnroadsRunner
from evolution.candidate import OutputParser


class EnROADSDataset(Dataset):
    """
    Torch dataset containing context.
    If we don't want a context we can pass None.
    """
    def __init__(self, context_df: pd.DataFrame):
        if context_df is None:
            self.X = torch.ones((1, 1))
            self.context_dicts = [[{}]]
        else:
            self.X = torch.tensor(context_df.to_numpy(), dtype=torch.float32)
            self.context_dicts = context_df.to_dict(orient="records")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.context_dicts[idx]


class EnROADSEvaluator(Evaluator):
    """
    Evaluates candidates by generating the actions and running the enroads model on them.
    """
    def __init__(self, context_path: Path, actions: list[str], outcomes: list[str], batch_size: int, device: str):
        super().__init__(outcomes)
        context_df = pd.read_csv(context_path) if context_path else None
        self.dataset = EnROADSDataset(context_df)
        self.batch_size = batch_size
        self.actions = actions
        self.outcomes = outcomes
        self.output_parser = OutputParser(actions, device)
        self.outcome_manager = OutcomeManager(outcomes)

        self.enroads_runner = EnroadsRunner()

    def update_predictor(self, elites):
        """
        We need to fill this in to make it not abstract.
        """
        return

    def validate_outcomes(self, outcomes_df: pd.DataFrame):
        """
        Ensures our outcome columns don't have NaNs or infs
        """
        outcome_keys = [key for key in self.outcomes if key in outcomes_df.columns]
        subset = outcomes_df[outcome_keys]
        assert not subset.isna().any().any(), "Outcomes contain NaNs."
        assert not np.isinf(subset.to_numpy()).any(), "Outcomes contain infs."
        return True

    def context_to_actions(self, candidate: EnROADSPrescriptor, dataset: Dataset) -> dict:
        """
        Takes a candidate EnROADSPrescriptor and a dataset of torch context, context_dict pairs and generates the
        actions for each context, then returns the context_actions dict to be passed into the EnROADS runner.
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        context_actions_dicts = []
        with torch.no_grad():
            for context_tensor, context_dicts in dataloader:
                # Get actions from context
                context_tensor = context_tensor.to(candidate.device)
                actions_tensor = candidate.forward(context_tensor)
                actions_dicts = self.output_parser.parse_actions_dicts(actions_tensor)

                for context_dict, actions_dict in zip(context_dicts, actions_dicts):
                    context_actions_dicts.append({**context_dict, **actions_dict})

        return context_actions_dicts

    def context_actions_to_outcomes(self, context_actions_dicts: list[dict[str, float]]) -> list[pd.DataFrame]:
        """
        Takes a list of context actions dicts and runs them through EnROADS to get a list of outcomes.
        """
        outcomes_dfs = []
        for context_actions_dict in context_actions_dicts:
            outcomes_df = self.enroads_runner.evaluate_actions(context_actions_dict)
            self.validate_outcomes(outcomes_df)
            outcomes_dfs.append(outcomes_df)
        return outcomes_dfs
    
    def cao_to_metrics(self, context_actions_dicts: list[dict], outcomes_dfs: list[pd.DataFrame]) -> dict:
        """
        Takes context actions dicts and outcomes dataframes and returns a dictionary of metrics.
        """
        total_metrics = {outcome: 0 for outcome in self.outcomes}
        for context_actions_dict, outcomes_df in zip(context_actions_dicts, outcomes_dfs):
            # Our context isn't part of the actions we took so we shouldn't evaluate our outcomes on them.
            actions_dict = {k: v for k, v in context_actions_dict.items() if k in self.actions}
            metrics = self.outcome_manager.process_outcomes(actions_dict, outcomes_df)
            for outcome in self.outcomes:
                total_metrics[outcome] += metrics[outcome]
        return {k: v / len(context_actions_dicts) for k, v in total_metrics.items()}

    def evaluate_candidate(self, candidate: EnROADSPrescriptor) -> np.ndarray:
        """
        Creates actions from the candidate, runs the enroads model, returns the outcomes.
        """
        # Get CAOs
        context_actions_dicts = self.context_to_actions(candidate, self.dataset)

        # Run En-ROADS
        outcomes_dfs = self.context_actions_to_outcomes(context_actions_dicts)

        # Get metrics from CAOs
        # Make sure metrics are in the right order
        metrics = self.cao_to_metrics(context_actions_dicts, outcomes_dfs)
        return np.array([metrics[outcome] for outcome in self.outcomes])
