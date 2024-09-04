"""
Government revenue outcome implementation.
"""
import pandas as pd

from evolution.outcomes.outcome import Outcome


class RevenueOutcome(Outcome):
    """
    Gets government spending.
    """
    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the total government net revenue from adjustments, or 0 if it is positive.
        We don't want to reward governments for making money off of the climate crisis.
        """
        revenue = outcomes_df["Government net revenue from adjustments"].sum()
        return min(revenue, 0)
