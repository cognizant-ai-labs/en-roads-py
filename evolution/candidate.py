"""
Candidate class to be used during evolution.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from enroadspy import load_input_specs


class Candidate():
    """
    Candidate class that holds the model and stores evaluation and sorting information for evolution.
    Model can be persisted to disk.
    """
    def __init__(self,
                 cand_id: str,
                 parents: list[str],
                 model_params: dict,
                 actions: list[str]):
        self.cand_id = cand_id
        self.actions = actions
        self.metrics = {}

        self.parents = parents
        self.rank = None
        self.distance = None

        # Model
        self.model_params = model_params
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NNPrescriptor(**model_params).to(self.device)
        self.model.eval()

        self.output_parser = OutputParser(actions, device=self.device)

    @classmethod
    def from_pymoo_params(cls, x: np.ndarray, model_params: dict, actions: list[str]):
        """
        Creates a candidate from a 1d numpy array of parameters that have to be reshaped into tensors and loaded
        as a state dict.
        """
        candidate = cls("pymoo", [], model_params, actions)

        flattened = torch.Tensor(x)
        state_dict = OrderedDict()
        pcount = 0

        in_size = model_params["in_size"]
        hidden_size = model_params["hidden_size"]
        out_size = model_params["out_size"]

        state_dict["nn.0.weight"] = flattened[:in_size * hidden_size].reshape(hidden_size, in_size)
        pcount += in_size * hidden_size

        state_dict["nn.0.bias"] = flattened[pcount:pcount + hidden_size]
        pcount += model_params["hidden_size"]

        state_dict["nn.2.weight"] = flattened[pcount:pcount + hidden_size * out_size].reshape(out_size, hidden_size)
        pcount += hidden_size * out_size

        state_dict["nn.2.bias"] = flattened[pcount:pcount + out_size]

        candidate.model.load_state_dict(state_dict)
        return candidate

    @classmethod
    def from_seed(cls, path: Path, model_params: dict, actions: list[str]):
        """
        Loads PyTorch seed from disk.
        """
        cand_id = path.stem
        parents = []
        candidate = cls(cand_id, parents, model_params, actions)
        candidate.model.load_state_dict(torch.load(path))
        return candidate

    def save(self, path: Path):
        """
        Saves PyTorch state dict to disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def prescribe(self, x: torch.Tensor) -> list[dict[str, float]]:
        """
        Runs the model on a batch of contexts and returns a list of actions dicts.
        Parses the output of our model so that we can use it in en-roads model.
        """
        with torch.no_grad():
            nn_outputs = self.model.forward(x)
            outputs = self.output_parser.parse_output(nn_outputs).cpu().numpy()
        actions_dicts = [dict(zip(self.actions, output.tolist())) for output in outputs]
        return actions_dicts

    def record_state(self):
        """
        Records metrics as well as seed and parents for reconstruction.
        """
        state = {
            "cand_id": self.cand_id,
            "parents": self.parents,
            "rank": self.rank,
            "distance": self.distance,
        }
        for metric, value in self.metrics.items():
            state[metric] = value
        return state

    def __str__(self):
        return f"Candidate({self.cand_id})"

    def __repr__(self):
        return f"Candidate({self.cand_id})"


class NNPrescriptor(torch.nn.Module):
    """
    Torch neural network that the candidate wraps around.
    """
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, out_size),
            torch.nn.Sigmoid()
        )

        # Orthogonal initialization
        for layer in self.nn:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, x):
        """
        Forward pass of neural network.
        Returns a tensor of shape (batch_size, num_actions).
        Values are scaled between 0 and 1.
        """
        nn_output = self.nn(x)
        return nn_output


class OutputParser():
    """
    Parses the output of our NNPrescriptor. All the values are between 0 and 1 and it's our job to scale them to
    match the input specs. It's ok to have an end date before a start date, the simulator just handles it interally.
    """
    def __init__(self, actions: list[str], device="cpu"):
        input_specs = load_input_specs()
        self.actions = actions

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
