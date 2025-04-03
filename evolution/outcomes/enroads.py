"""
Basic enroads outcome
"""
import pandas as pd

from enroadspy import load_input_specs
from evolution.outcomes.outcome import Outcome


class EnroadsOutcome(Outcome):
    """
    Just checks the final year of an enroads outcome column.
    """
    def __init__(self, outcome: str):
        # Check that this outcome is actually in En-ROADS. Otherwise, raise an error.
        input_specs = load_input_specs()
        if outcome not in input_specs["varId"].unique():
            raise ValueError(f"Invalid EnroadsOutcome: {outcome} not in input specs")

        self.outcome = outcome

    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Simple case where we return the specified outcome in 2100.
        """
        return outcomes_df[self.outcome].iloc[-1]
