"""
This outcome is not used in our evolution but rather is how we get the correct slider values to match the ar6 ssps.
"""
import pandas as pd

from evolution.outcomes.outcome import Outcome


class GDPOutcome(Outcome):
    """
    Outcome used in finding the context values to match the scenarios in the AR6 database for the 5 scenarios.
    """
    def __init__(self, scenario: int):
        assert scenario >= 1 and scenario <= 5, "Scenario must be between 1 and 5."
        year_cols = [2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
        year_cols_str = [str(year) for year in year_cols]
        self.year_idxs = [year - 1990 for year in year_cols]

        df = pd.read_csv("experiments/scenarios/ar6_snapshot_1723566520.csv/ar6_snapshot_1723566520.csv")
        df = df.dropna(subset=["Scenario"])
        df = df[df["Scenario"] == f"SSP{scenario}-Baseline"]
        df = df.dropna(axis=1)
        gdp_df = df[df["Variable"] == "GDP|PPP"]
        self.label_gdp = gdp_df[year_cols_str].values[0]

    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the max cost of energy.
        """
        gdp_col = outcomes_df["Global GDP"]
        sim_gdp = (gdp_col.iloc[self.year_idxs] * 1000).values
        mse = ((sim_gdp - self.label_gdp) ** 2).mean()
        return mse
