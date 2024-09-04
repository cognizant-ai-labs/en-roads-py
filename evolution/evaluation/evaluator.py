"""
Evaluates candidates in order for them to be sorted.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evolution.candidate import Candidate
from evolution.evaluation.data import ContextDataset
from evolution.outcomes.outcome_manager import OutcomeManager
from enroadspy import load_input_specs
from enroadspy.enroads_runner import EnroadsRunner


class Evaluator:
    """
    Evaluates candidates by generating the actions and running the enroads model on them.
    Generates and stores context data based on config using ContextDataset.
    """
    def __init__(self, context: list[str], actions: list[str], outcomes: dict[str, bool]):
        self.actions = actions
        self.outcomes = outcomes
        self.outcome_manager = OutcomeManager(outcomes)

        # Precise float is required to load the enroads inputs properly
        self.input_specs = load_input_specs()

        self.context = context
        # Context Dataset outputs a scaled tensor and nonscaled tensor. The scaled tensor goes into PyTorch and
        # the nonscaled tensor is used to reconstruct the context that goes into enroads.
        self.context_dataset = ContextDataset(context)
        self.context_dataloader = DataLoader(self.context_dataset, batch_size=3, shuffle=False)
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        self.enroads_runner = EnroadsRunner()

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

    def evaluate_candidate(self, candidate: Candidate):
        """
        Evaluates a single candidate by running all the context through it and receiving all the batches of actions.
        Then evaluates all the actions and returns the average outcome.
        """
        outcomes_dfs = []
        cand_results = []
        # Iterate over batches of contexts
        for batch_tensor, batch_context in self.context_dataloader:
            context_dicts = self.reconstruct_context_dicts(batch_context)
            actions_dicts = candidate.prescribe(batch_tensor.to(self.device))
            for actions_dict, context_dict in zip(actions_dicts, context_dicts):
                # Add context to actions so we can pass it into the model
                actions_dict.update(context_dict)
                outcomes_df = self.enroads_runner.evaluate_actions(actions_dict)
                self.validate_outcomes(outcomes_df)
                outcomes_dfs.append(outcomes_df)
                cand_results.append(self.outcome_manager.process_outcomes(actions_dict, outcomes_df))

        candidate.metrics = {key: np.mean([result[key] for result in cand_results]) for key in cand_results[0]}
        return outcomes_dfs

    def evaluate_candidates(self, candidates: list[Candidate]):
        """
        Evaluates all candidates. Doesn't unnecessarily evaluate candidates that have already been evaluated.
        """
        for candidate in tqdm(candidates, leave=False):
            if len(candidate.metrics) == 0:
                self.evaluate_candidate(candidate)
