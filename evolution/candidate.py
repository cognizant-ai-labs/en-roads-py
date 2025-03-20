"""
Candidate class to be used during evolution.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd
from presp.prescriptor import NNPrescriptor
import torch

from enroadspy import load_input_specs


class EnROADSPrescriptor(NNPrescriptor):
    """
    Prescriptor candidate for En-ROADS. Runs context as a torch tensor through the model and parses the output into
    the enroadsrunner actions dict format
    """
    def __init__(self, model_params: list[dict], actions: list[str], device: str = "cpu"):
        super().__init__(model_params, device)
        self.actions = list(actions)
        self.output_parser = OutputParser(self.actions, device=device)

    def forward(self, context: torch.Tensor) -> list[dict]:
        with torch.no_grad():
            nn_outputs = super().forward(context)
            outputs = self.output_parser.parse_output(nn_outputs).cpu().numpy()
        actions_dicts = [dict(zip(self.actions, output.tolist())) for output in outputs]
        return actions_dicts

    @classmethod
    def from_pymoo_params(cls, x: np.ndarray, model_params: dict, actions: list[str]) -> "EnROADSPrescriptor":
        """
        Creates a candidate from a 1d numpy array of parameters that have to be reshaped into tensors and loaded
        as a state dict.
        """
        candidate = cls(model_params, actions)

        flattened = torch.Tensor(x)
        state_dict = OrderedDict()
        pcount = 0

        pcount = 0
        for i, layer in enumerate(model_params):
            if layer["type"] == "linear":
                in_size = layer["in_features"]
                out_size = layer["out_features"]

                state_dict[f"{i}.weight"] = flattened[pcount:pcount+(in_size*out_size)].reshape(out_size, in_size)
                pcount += in_size * out_size
                state_dict[f"{i}.bias"] = flattened[pcount:pcount+out_size]
                pcount += out_size

        candidate.model.load_state_dict(state_dict)
        return candidate


class OutputParser():
    """
    Parses the output of our neural network. All the values are between 0 and 1 and it's our job to scale them to
    match the input specs. It's ok to have an end date before a start date, the simulator just handles it interally.
    """
    def __init__(self, actions: list[str], device: str = "cpu"):
        input_specs = load_input_specs()
        self.actions = actions

        # Make sure all the actions are in the input specs
        valid_actions = input_specs["varId"].unique()
        for action in actions:
            if action not in valid_actions:
                raise ValueError(f"Action {action} not in input specs")

        # Index into the dataframe with actions
        filtered = input_specs[input_specs["varId"].isin(actions)].copy()
        filtered["varId"] = pd.Categorical(filtered["varId"], categories=actions, ordered=True)
        filtered = filtered.sort_values("varId")

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
