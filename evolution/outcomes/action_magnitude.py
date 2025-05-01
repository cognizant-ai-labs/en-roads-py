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
                scaling_values[row["id"]] = [row["minValue"], row["defaultValue"], row["maxValue"]]
            else:
                # We don't count change in switches
                scaling_values[row["id"]] = [0, 0, 0]
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
