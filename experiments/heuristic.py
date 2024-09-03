"""
Comparing our evolution results to a greedy heuristic.
"""
import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from evolution.outcomes.enroads import EnroadsOutcome
from enroadspy import load_input_specs
from enroadspy.enroads_runner import EnroadsRunner
from enroadspy.generate_url import actions_to_url


class Heuristic:
    """
    Finds the best action by maxing or minning every action and taking the best one.
    We can also generate a plot of these results to visualize which actions are most important greedily.
    """
    def __init__(self, actions: list[str]):
        self.input_specs = load_input_specs()
        self.runner = EnroadsRunner()
        self.outcome_parser = EnroadsOutcome("CO2 Equivalent Net Emissions")
        self.actions = list(actions)

    def check_action_values(self, actions_dict: dict[str, float], action: str) -> tuple[float, float]:
        """
        Takes an action and sees if the max or the min is better. Then returns either the max or min and the resulting
        value.
        """
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
        """
        Finds the best actions greedily by going over each action we haven't used left, looking at if its max or 
        min value is the best, then adding it if so.
        """
        action_order = []
        actions_dict = {}

        actions_left = list(self.actions)
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
        """
        Plot our actions used in a nice grid. This will form a staircase ideally that shows the actions used.
        """
        grid = []

        for i, action in enumerate(action_order):
            val = actions_dict[action]
            row = np.zeros(len(action_order))
            max_value = self.input_specs[self.input_specs["varId"] == action].iloc[0]["maxValue"]
            on_value = self.input_specs[self.input_specs["varId"] == action].iloc[0]["onValue"]
            if val in (max_value, on_value):
                row[:i+1] = 1
            else:
                row[:i+1] = -1
            grid.append(row)

        action_labels = []
        for action in action_order:
            action_labels.append(self.input_specs[self.input_specs["varId"] == action]["varName"].iloc[0])

        grid = np.stack(grid).T
        grid = np.flip(grid, axis=0)
        plt.figure(figsize=(9, 9))
        plt.yticks(range(len(action_labels)), reversed(action_labels))
        plt.xticks(range(len(action_labels)), rotation=90)
        plt.title("Greedy Heuristic Actions Used")
        plt.imshow(grid, cmap=ListedColormap(["lightgreen", "white", "green"]))
        plt.show()

    def get_heuristic_urls(self, action_order: list[str], actions_dict: dict[str, float]) -> list[str]:
        """
        Get the URL adding each sequential action from the action order.
        """
        urls = []
        for i in range(len(action_order)):
            url_dict = {action: actions_dict[action] for action in action_order[:i+1]}
            urls.append(actions_to_url(url_dict))
        return urls


def main():
    """
    Main method to run and plot our heuristics.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    actions = config["actions"]

    heuristic = Heuristic(actions)
    action_order, actions_dict = heuristic.find_heuristic()
    heuristic.plot_actions_used(action_order, actions_dict)


if __name__ == "__main__":
    main()
