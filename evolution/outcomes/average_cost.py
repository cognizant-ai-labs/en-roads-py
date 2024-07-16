import pandas as pd

from evolution.outcomes.outcome import Outcome

class AverageCostOutcome(Outcome):
    def process_outcomes(self, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the average cost from 2025 onwards.
        """
        cost_col = outcomes_df["Adjusted cost of energy per GJ"]
        cost = cost_col.iloc[2025-1990:].mean()
        return cost