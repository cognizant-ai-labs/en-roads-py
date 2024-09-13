"""
Converts the title of an outcome from the config to the corresponding outcome object that processes the outcomes_df and
actions_dict.
"""
import pandas as pd

from evolution.outcomes.actions import ActionsOutcome
from evolution.outcomes.action_magnitude import ActionMagnitudeOutcome
from evolution.outcomes.average_cost import AverageCostOutcome
from evolution.outcomes.cost_change_year import CostChangeYearOutcome
from evolution.outcomes.cost_exp import CostExpOutcome
from evolution.outcomes.cost_ssd import CostSSDOutcome
from evolution.outcomes.energy_change import EnergyChangeOutcome
from evolution.outcomes.energy_change_exp import EnergyChangeExpOutcome
from evolution.outcomes.energy_change_ssd import EnergyChangeSSDOutcome
from evolution.outcomes.enroads import EnroadsOutcome
from evolution.outcomes.gdp_mse import GDPOutcome
from evolution.outcomes.max_cost import MaxCostOutcome
from evolution.outcomes.near_cost import NearCostOutcome
from evolution.outcomes.paris_relax import ParisRelaxOutcome
from evolution.outcomes.paris import ParisOutcome
from evolution.outcomes.revenue import RevenueOutcome
from evolution.outcomes.total_energy import TotalEnergyOutcome
from evolution.outcomes.zero_emissions import ZeroEmissionsOutcome


class OutcomeManager():
    """
    Manages many outcomes at once for the evaluator.
    """
    def __init__(self, outcomes: list[str]):
        outcome_dict = {}
        for outcome in outcomes:
            if outcome == "Actions taken":
                outcome_dict[outcome] = ActionsOutcome()
            elif outcome == "Action magnitude":
                outcome_dict[outcome] = ActionMagnitudeOutcome()
            elif outcome == "Average Adjusted cost of energy per GJ":
                outcome_dict[outcome] = AverageCostOutcome()
            elif outcome == "Average Percent Energy Change":
                outcome_dict[outcome] = EnergyChangeOutcome()
            elif outcome == "Cost of energy next 10 years":
                outcome_dict[outcome] = NearCostOutcome()
            elif outcome == "Year Zero Emissions Reached":
                outcome_dict[outcome] = ZeroEmissionsOutcome()
            elif outcome == "Government net revenue below zero":
                outcome_dict[outcome] = RevenueOutcome()
            elif outcome == "Emissions Above Paris Agreement":
                outcome_dict[outcome] = ParisOutcome()
            elif outcome == "Max cost of energy":
                outcome_dict[outcome] = MaxCostOutcome()
            elif outcome == "Cost change year":
                outcome_dict[outcome] = CostChangeYearOutcome()
            elif outcome == "Total energy below baseline":
                outcome_dict[outcome] = TotalEnergyOutcome()
            elif outcome == "Temperature above 1.5C":
                outcome_dict[outcome] = ParisRelaxOutcome()
            elif outcome == "Cost Change Squared":
                outcome_dict[outcome] = CostSSDOutcome()
            elif outcome == "Energy Change Squared":
                outcome_dict[outcome] = EnergyChangeSSDOutcome()
            elif outcome == "Cost Change Exponential":
                outcome_dict[outcome] = CostExpOutcome()
            elif outcome == "Energy Change Exponential":
                outcome_dict[outcome] = EnergyChangeExpOutcome()
            elif "GDP MSE" in outcome:
                scenario = int(outcome.split(" ")[-1])
                outcome_dict[outcome] = GDPOutcome(scenario)
            else:
                outcome_dict[outcome] = EnroadsOutcome(outcome)

        self.outcome_dict = outcome_dict

    def process_outcomes(self, actions_dict: dict[str, float], outcomes_df: pd.DataFrame) -> dict[str, float]:
        """
        Processes outcomes from outcomes_df with all outcomes in outcome_dict.
        """
        results_dict = {}
        for outcome, outcome_obj in self.outcome_dict.items():
            results_dict[outcome] = outcome_obj.process_outcomes(actions_dict, outcomes_df)

        return results_dict
