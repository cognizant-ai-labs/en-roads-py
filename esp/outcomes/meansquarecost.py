import pandas as pd

from esp.outcomes.outcome import Outcome


class MeanSquareCost(Outcome):
    """
    Takes the mean of the square of the cost of energy.
    """
    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        cost_col = outcomes_df["Adjusted cost of energy per GJ"]
        return cost_col.pow(2).mean()