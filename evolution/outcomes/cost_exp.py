"""
Outcome that sums the exponent of positive difference of costs between each year.
"""
import numpy as np
import pandas as pd

from evolution.outcomes.outcome import Outcome


class CostExpOutcome(Outcome):
    """
    Cost SSD Outcome
    """
    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Sums the square difference of costs between each year.
        """
        cost_col = outcomes_df["Adjusted cost of energy per GJ"].iloc[2024-1990:]
        change = cost_col.diff().shift(-1).dropna()
        ssd = np.exp(change[change > 0]).sum()
        return ssd
