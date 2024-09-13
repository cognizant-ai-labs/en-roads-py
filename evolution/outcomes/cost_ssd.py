"""
Outcome that sums the square positive difference of costs between each year.
"""
import pandas as pd

from evolution.outcomes.outcome import Outcome


class CostSSDOutcome(Outcome):
    """
    Cost SSD Outcome
    """
    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Sums the square difference of costs between each year.
        """
        cost_col = outcomes_df["Adjusted cost of energy per GJ"].iloc[2024-1990:]
        change = cost_col.diff().shift(-1).dropna()
        ssd = (change[change > 0] ** 2).sum()
        return ssd
