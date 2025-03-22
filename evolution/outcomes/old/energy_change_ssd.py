"""
Energy change SSD
"""
import pandas as pd

from evolution.outcomes.outcome import Outcome


class EnergyChangeSSDOutcome(Outcome):
    """
    Takes the sum of sum of squared differences of energy demand changes *FLAT* vs. percentage.
    """
    def __init__(self):
        # We don't want fossil fuels because it double counts
        energies = ["bio", "coal", "gas", "oil", "renew and hydro", "new tech", "nuclear"]
        self.demands = [f"Primary energy demand of {energy}" for energy in energies]

    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the percent the distribution changed in a given year averaged over the whole outcomes_df.
        """
        energy_change = outcomes_df[self.demands].diff().shift(-1).fillna(0)  # Shift so we align with the year
        square_diff = energy_change ** 2
        ssd = square_diff.sum().sum()
        return ssd
