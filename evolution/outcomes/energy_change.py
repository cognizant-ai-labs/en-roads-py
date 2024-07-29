import pandas as pd

from evolution.outcomes.outcome import Outcome


class EnergyChangeOutcome(Outcome):
    def __init__(self):
        # We don't want fossil fuels because it double counts
        energies = ["bio", "coal", "gas", "oil", "renew and hydro", "new tech", "nuclear"]
        self.demands = [f"Primary energy demand of {energy}" for energy in energies]

    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the percent the distribution changed in a given year averaged over the whole outcomes_df.
        """
        energy_change = outcomes_df[self.demands].diff().shift(-1).fillna(0)  # Shift so we align with the year
        row_sums = outcomes_df[self.demands].sum(axis=1)
        percent_change = energy_change.div(row_sums, axis=0)
        percent_change_sum = percent_change.abs().sum(axis=1)
        average_percent_change = percent_change_sum.mean()
        return average_percent_change
