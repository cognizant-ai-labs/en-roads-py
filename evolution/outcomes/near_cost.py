import pandas as pd

from evolution.outcomes.outcome import Outcome

class NearCostOutcome(Outcome):
    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the average cost of energy over the next 10 years.
        """
        cost_col = outcomes_df["Adjusted cost of energy per GJ"]
        cost = cost_col.iloc[2025-1990:2035-1990].mean()
        return cost