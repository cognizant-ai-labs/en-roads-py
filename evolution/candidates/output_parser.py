"""
Output Parser class that goes from normalized torch tensors between 0 and 1 to En-ROADS actions.
"""
import pandas as pd
import torch

from enroadspy import load_input_specs, id_to_name


class OutputParser():
    """
    Parses the output of our neural network. All the values are between 0 and 1 and it's our job to scale them to
    match the input specs.
    NOTE: It's ok to have an end date before a start date, the simulator just handles it internally.
    TODO: Make it not ok to have an end date before a start date, these count as actions taken for the actions taken
    count even though they shouldn't.
    """
    def __init__(self, actions: list[int], device: str = "cpu"):
        input_specs = load_input_specs()
        self.actions = actions

        # Make sure all the actions are in the input specs
        valid_actions = input_specs["id"].unique()
        for action in actions:
            if action not in valid_actions:
                raise ValueError(f"Action {action}, {id_to_name(action, input_specs)} not in input specs")

        # Index into the dataframe with actions
        filtered = input_specs[input_specs["id"].isin(actions)].copy()
        filtered["id"] = pd.Categorical(filtered["id"], categories=actions, ordered=True)
        filtered = filtered.sort_values("id")

        # Non-sliders are left scaled between 0 and 1.
        bias = filtered["minValue"].fillna(0).values
        scale = filtered["maxValue"].fillna(1).values - filtered["minValue"].fillna(0).values

        # Keep track of which actions are switches so we can snap them to their on or off values.
        switches = (filtered["kind"] == "switch").values
        on_values = filtered["onValue"].fillna(0).values
        off_values = filtered["offValue"].fillna(0).values

        # Torch values to scale by
        self.bias = torch.FloatTensor(bias).to(device)
        self.scale = torch.FloatTensor(scale).to(device)

        self.switches = torch.BoolTensor(switches).to(device)
        self.on_values = torch.FloatTensor(on_values).to(device)
        self.off_values = torch.FloatTensor(off_values).to(device)

        self.device = device

    def parse_output(self, nn_outputs: torch.Tensor) -> torch.Tensor:
        """
        nn_outputs: (batch_size, num_actions)
        Does the actual parsing of the outputs.
        Scale the sliders by multiplying by scale then adding bias.
        Snap switches to on or off based on the output.
        """
        nn_outputs = nn_outputs.to(self.device)
        b = nn_outputs.shape[0]
        # First we scale our sliders
        scaled = nn_outputs * self.scale + self.bias

        # Now we snap our switches to on or off
        scaled[:, self.switches] = torch.where(scaled[:, self.switches] > 0.5,
                                               self.on_values.repeat(b, 1)[:, self.switches],
                                               self.off_values.repeat(b, 1)[:, self.switches])

        return scaled

    def unparse(self, parsed_outputs: torch.Tensor) -> torch.Tensor:
        """
        Undo the parsing performed in parse_output for seed training.
        """
        parsed_outputs = parsed_outputs.to(self.device)
        b = parsed_outputs.shape[0]

        # Unscale sliders
        unscaled = (parsed_outputs - self.bias) / self.scale

        # Undo switch snapping
        on_values = self.on_values.repeat(b, 1)[:, self.switches]
        unscaled[:, self.switches] = torch.where(unscaled[:, self.switches] == on_values, 1, 0).float()

        return unscaled
