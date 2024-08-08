import pandas as pd

from evolution.outcomes.outcome import Outcome


class CostChangeYearOutcome(Outcome):
    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the first year the cost of energy changes by more than a given threshold (2 default)
        """
        cost_col = outcomes_df["Adjusted cost of energy per GJ"].iloc[2024-1990:]
        # pct_change = (cost_col.shift(-1) / cost_col).dropna() - 1
        # greater_years = pct_change[pct_change > 0.1]
        change = cost_col.diff().shift(-1).dropna()
        greater_years = change[change > 2]
        # Return 2101 if no year is found to show this is the best case scenario
        if greater_years.empty:
            year = 2101
        else:
            year = greater_years.index[0] + 1990
        return year