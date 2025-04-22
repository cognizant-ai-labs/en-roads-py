"""
Outcome to see the difference between the actions taken and the default actions.
"""
from evolution.outcomes.outcome import Outcome

from enroadspy import load_input_specs


class ActionMagnitudeOutcome(Outcome):
    """
    Get the normalized difference between our action and the default.
    This assumes the range of the sliders is considered a "reasonable" range of actions.
    """
    # pylint: disable=no-member
    def __init__(self):
        input_specs = load_input_specs()
        scaling_values = {}
        for _, row in input_specs.iterrows():
            if row["kind"] == "slider":
                scaling_values[row["varId"]] = [row["minValue"], row["defaultValue"], row["maxValue"]]
            else:
                # We don't count change in switches
                scaling_values[row["varId"]] = [0, 0, 0]
        self.scaling_values = scaling_values

    def process_outcomes(self, actions_dict: dict[str, float], _) -> float:
        """
        Returns the magnitude of actions taken by normalizing their difference.
        """
        action_magnitude = 0
        for action, value in actions_dict.items():
            min_val, default_val, max_val = self.scaling_values[action]
            if max_val != min_val:
                action_magnitude += abs((value - default_val) / (max_val - min_val))

        return action_magnitude


class AverageActionMagnitudeOutcome(ActionMagnitudeOutcome):
    """
    Takes the average action magnitude across all actions rather than the sum.
    """
    def process_outcomes(self, actions_dict: dict[str, float], _) -> float:
        """
        Divides by the length of our actions dict this time.
        """
        magnitude = super().process_outcomes(actions_dict, _)
        return magnitude / len(actions_dict)


class SimpleActionMagnitudeOutcome(Outcome):
    """
    Uses the rangeDividers attribute in inputSpecs and counts how many dividers the action is from the default.
    """
    def __init__(self):
        input_specs = load_input_specs()
        default_values = {}
        low_dividers = {}
        high_dividers = {}

        for _, row in input_specs.iterrows():
            # Check if range dividers is not none, is a list, and has a length > 0
            if row["rangeDividers"] is not None and isinstance(row["rangeDividers"], list) and len(row["rangeDividers"]) > 0:
                default_values[row["varId"]] = row["defaultValue"]
                dividers = row["rangeDividers"]
                low_dividers[row["varId"]] = [d for d in dividers if d < row["defaultValue"]]
                high_dividers[row["varId"]] = [d for d in dividers if d > row["defaultValue"]]

        self.default_values = default_values
        self.low_dividers = low_dividers
        self.high_dividers = high_dividers

    def process_outcomes(self, actions_dict: dict[str, float], _) -> float:
        """
        Looks at each action and counts how many dividers it is away from the default, then averages the final number.
        """
        action_magnitude = 0
        for action, value in actions_dict.items():
            if action not in self.default_values:
                raise ValueError(f"Action {action} not a simple action!")

            default_val = self.default_values[action]
            low_dividers = self.low_dividers[action]
            high_dividers = self.high_dividers[action]

            # Add one to this count because the value is not default
            if value < default_val:
                action_magnitude += len([d for d in low_dividers if d > value]) + 1
            elif value > default_val:
                action_magnitude += len([d for d in high_dividers if d < value]) + 1

        action_magnitude /= len(actions_dict)

        return action_magnitude
