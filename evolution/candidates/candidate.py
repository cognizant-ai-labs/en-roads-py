"""
Candidate class to be used during evolution.
"""
from collections import OrderedDict

import numpy as np
from presp.prescriptor import NNPrescriptor
import torch

from evolution.candidates.output_parser import OutputParser


class EnROADSPrescriptor(NNPrescriptor):
    """
    Prescriptor candidate for En-ROADS. Runs context as a torch tensor through the model and parses the output into
    the enroadsrunner actions dict format
    """
    def __init__(self, model_params: list[dict], actions: list[int], device: str = "cpu"):
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
    def from_pymoo_params(cls, x: np.ndarray, model_params: dict, actions: list[int]) -> "EnROADSPrescriptor":
        """
        Creates a candidate from a 1d numpy array of parameters that have to be reshaped into tensors and loaded
        as a state dict.
        """
        candidate = cls(model_params, actions)

        flattened = torch.Tensor(x)
        state_dict = OrderedDict()
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
