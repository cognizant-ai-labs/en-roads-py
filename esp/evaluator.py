"""
Evaluates candidates in order for them to be sorted.
"""
import numpy as np
import pandas as pd
from presp.evaluator import Evaluator
import torch

from esp.prescriptor import EnROADSPrescriptor
from esp.outcome_manager import OutcomeManager
from enroadspy import load_input_specs
from enroadspy.enroads_runner import EnroadsRunner


class EnROADSEvaluator(Evaluator):
    """
    Evaluates candidates by generating the actions and running the enroads model on them.
    """
    def __init__(self, outcomes: list[str]):
        super().__init__(outcomes)
        self.outcomes = outcomes
        self.outcome_manager = OutcomeManager(outcomes)

        self.input_specs = load_input_specs()

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

    def evaluate_candidate(self, candidate: EnROADSPrescriptor) -> np.array:
        """
        Creates actions from the candidate, runs the enroads model, returns the outcomes.
        """
        with torch.no_grad():
            actions_dict = candidate.forward(None)

        outcomes_df = self.enroads_runner.evaluate_actions(actions_dict)
        self.validate_outcomes(outcomes_df)
        metrics = self.outcome_manager.process_outcomes(actions_dict, outcomes_df)
        return np.array([metrics[outcome] for outcome in self.outcomes])
