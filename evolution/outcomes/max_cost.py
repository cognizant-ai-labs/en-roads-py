"""
Max cost outcome implementation.
"""
import pandas as pd

from evolution.outcomes.outcome import Outcome


class MaxCostOutcome(Outcome):
    """
    Gets the max cost over the entire range of years.
    """
    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the max cost of energy.
        """
        cost_col = outcomes_df["Adjusted cost of energy per GJ"]
        cost = cost_col.max()
        return cost
