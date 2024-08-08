from pathlib import Path

import pandas as pd

from evolution.outcomes.outcome import Outcome
from enroads_runner import EnroadsRunner


class TotalEnergyOutcome(Outcome):

    def __init__(self):
        # TODO: Can this cause a race condition?
        temp_dir = Path("evolution/outcomes/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        runner = EnroadsRunner(temp_dir=str(temp_dir))
        baseline_df = runner.evaluate_actions({})
        self.baseline_energy = baseline_df["Total Primary Energy Demand"]

    def process_outcomes(self, _, outcomes_df: pd.DataFrame) -> float:
        """
        Returns the total energy below the baseline.
        """
        total_energy = outcomes_df["Total Primary Energy Demand"]
        energy_diff = total_energy[total_energy < self.baseline_energy] - \
            self.baseline_energy[total_energy < self.baseline_energy]
        return energy_diff.sum()