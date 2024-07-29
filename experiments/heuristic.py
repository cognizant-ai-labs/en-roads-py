
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from evolution.outcomes.enroads import EnroadsOutcome
from enroads_runner import EnroadsRunner


class Heuristic:

    def __init__(self, actions: list[str]):
        self.input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        self.runner = EnroadsRunner("experiments/temp")
        self.outcome_parser = EnroadsOutcome("CO2 Equivalent Net Emissions")
        self.actions = actions

    def check_action_values(self, actions_dict: dict[str, float], action: str) -> tuple[float, float]:
        row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
        if row["kind"] == "switch":
            possibilities = [row["onValue"], row["offValue"]]
        else:
            possibilities = [row["minValue"], row["maxValue"]]

        best_value = None
        best_outcome = None
        for possibility in possibilities:
            actions_dict[action] = possibility
            outcomes_df = self.runner.evaluate_actions(actions_dict)
            outcome = self.outcome_parser.process_outcomes(actions_dict, outcomes_df)
            if best_outcome is None or outcome < best_outcome:
                best_outcome = outcome
                best_value = possibility

        actions_dict.pop(action)
        return best_value, best_outcome

    # pylint: disable=no-member
    def find_heuristic(self) -> tuple[list[str], dict[str, float]]:
        action_order = []
        actions_dict = {}

        actions_left = self.actions
        while len(actions_left) > 0:
            best_action = None
            best_action_outcome = None
            best_action_value = None
            for action in actions_left:
                best_value, best_outcome = self.check_action_values(actions_dict, action)
                if best_action_outcome is None or best_outcome < best_action_outcome:
                    best_action_outcome = best_outcome
                    best_action = action
                    best_action_value = best_value

            actions_dict[best_action] = best_action_value
            action_order.append(best_action)
            actions_left.remove(best_action)

        return action_order, actions_dict
    
    # pylint: enable=no-member

    def plot_actions_used(self, action_order: list[str], actions_dict: dict[str, float]):
        grid = []
        
        for i, action in enumerate(action_order):
            val = actions_dict[action]
            row = np.zeros(len(action_order))
            max_value = self.input_specs[self.input_specs["varId"] == action].iloc[0]["maxValue"]
            on_value = self.input_specs[self.input_specs["varId"] == action].iloc[0]["onValue"]
            if val == max_value or val == on_value:
                row[:i+1] = 1
            else:
                row[:i+1] = -1
            grid.append(row)

        action_labels = []
        for action in action_order:
            action_labels.append(self.input_specs[self.input_specs["varId"] == action]["varName"].iloc[0])

        grid = np.stack(grid).T
        grid = np.flip(grid, axis=0)
        plt.figure(figsize=(9,9))
        plt.yticks(range(len(action_labels)), reversed(action_labels))
        plt.xticks(range(len(action_labels)), rotation=90)
        plt.title("Greedy Heuristic Actions Used")
        plt.imshow(grid, cmap=ListedColormap(["lightgreen", "white", "green"]))
        plt.show()


def main():
    actions = [
        "_source_subsidy_delivered_coal_tce",
        "_source_subsidy_start_time_delivered_coal",
        "_source_subsidy_stop_time_delivered_coal",
        "_no_new_coal",
        "_year_of_no_new_capacity_coal",
        "_utilization_adjustment_factor_delivered_coal",
        "_utilization_policy_start_time_delivered_coal",
        "_utilization_policy_stop_time_delivered_coal",
        "_target_accelerated_retirement_rate_electric_coal",
        "_source_subsidy_delivered_oil_boe",
        "_source_subsidy_start_time_delivered_oil",
        "_source_subsidy_stop_time_delivered_oil",
        "_no_new_oil",
        "_year_of_no_new_capacity_oil",
        "_utilization_adjustment_factor_delivered_oil",
        "_utilization_policy_start_time_delivered_oil",
        "_utilization_policy_stop_time_delivered_oil",
        "_source_subsidy_delivered_gas_mcf",
        "_source_subsidy_start_time_delivered_gas",
        "_source_subsidy_stop_time_delivered_gas",
        "_no_new_gas",
        "_year_of_no_new_capacity_gas",
        "_utilization_adjustment_factor_delivered_gas",
        "_utilization_policy_start_time_delivered_gas",
        "_utilization_policy_stop_time_delivered_gas",
        "_source_subsidy_renewables_kwh",
        "_source_subsidy_start_time_renewables",
        "_source_subsidy_stop_time_renewables",
        "_use_subsidies_by_feedstock",
        "_source_subsidy_delivered_bio_boe",
        "_source_subsidy_start_time_delivered_bio",
        "_source_subsidy_stop_time_delivered_bio",
        "_no_new_bio",
        "_year_of_no_new_capacity_bio",
        "_wood_feedstock_subsidy_boe",
        "_crop_feedstock_subsidy_boe",
        "_other_feedstock_subsidy_boe",
        "_source_subsidy_nuclear_kwh",
        "_source_subsidy_start_time_nuclear",
        "_source_subsidy_stop_time_nuclear",
        "_carbon_tax_initial_target",
        "_carbon_tax_phase_1_start",
        "_carbon_tax_time_to_achieve_initial_target",
        "_carbon_tax_final_target",
        "_carbon_tax_phase_3_start",
        "_carbon_tax_time_to_achieve_final_target",
        "_apply_carbon_tax_to_biofuels",
        "_ccs_carbon_tax_qualifier",
        "_qualifying_path_renewables",
        "_qualifying_path_nuclear",
        "_qualifying_path_new_zero_carbon",
        "_qualifying_path_beccs",
        "_qualifying_path_bioenergy",
        "_qualifying_path_fossil_ccs",
        "_qualifying_path_gas",
        "_electric_standard_active",
        "_electric_standard_target",
        "_electric_standard_start_year",
        "_electric_standard_target_time",
        "_emissions_performance_standard",
        "_performance_standard_time"
    ]
    heuristic = Heuristic(actions)
    action_order, actions_dict = heuristic.find_heuristic()
    heuristic.plot_actions_used(action_order, actions_dict)


if __name__ == "__main__":
    main()
