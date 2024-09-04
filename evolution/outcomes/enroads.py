"""
Basic enroads outcome
"""
import pandas as pd

from evolution.outcomes.outcome import Outcome


class EnroadsOutcome(Outcome):
    """
    Just checks the final year of an enroads outcome column.
    """
    def __init__(self, outcome: str):
        self.outcome = outcome

    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Simple case where we return the specified outcome in 2100.
        """
        return outcomes_df[self.outcome].iloc[-1]
