from pathlib import Path

import torch

class Candidate():
    """
    Candidate class that points to a model on disk.
    """
    def __init__(self, cand_id: str, parents: list[str], model_params: dict, outcomes):
        self.cand_id = cand_id
        self.outcomes = outcomes
        self.metrics = {}

        self.parents = parents
        self.rank = None
        self.distance = None

        # Model
        self.model_params = model_params
        self.model = NNPrescriptor(**model_params).to("mps")
        self.model.eval()

    @classmethod
    def from_seed(cls, path: Path, model_params: dict, outcomes):
        cand_id = path.stem
        parents = []
        candidate = cls(cand_id, parents, model_params, outcomes)
        candidate.model.load_state_dict(torch.load(path))
        return candidate
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def prescribe(self, x: torch.Tensor) -> dict:
        """
        Parses the output of our model so that we can use it in the AquaCrop model.
        """
        with torch.no_grad():
            outputs = self.model.forward(x).detach().cpu().tolist()
        return outputs

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
        for outcome in self.outcomes:
            state[outcome] = self.metrics[outcome]
        return state

    def __str__(self):
        return f"Candidate({self.cand_id})"
    
    def __repr__(self):
        return f"Candidate({self.cand_id})"

class NNPrescriptor(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, out_size),
            torch.nn.Sigmoid()
        )
        # TODO: fix hard coding
        self.bias = torch.tensor([-15, 2024, 2024, 0, 2024, 0, 2024, 2024, 0])
        self.scaler = torch.tensor([115, 76, 76, 100, 76, 100, 76, 76, 10])

    # pylint: disable=consider-using-enumerate
    def forward(self, x):
        nn_output = self.nn(x)
        self.scaler = self.scaler.to(x.device)
        self.bias = self.bias.to(x.device)
        for i in range(len(nn_output)):
            # Handle start/stop dates
            if i not in [2, 7]:
                nn_output[i] = nn_output[i] * self.scaler[i] + self.bias[i]
            else:
                nn_output[i] = nn_output[i] * (self.scaler[i] - (nn_output[i-1] - self.bias[i-1])) + nn_output[i-1]
        assert nn_output[2] > nn_output[1], f"Start date is after end date. Start: {nn_output[1]}, End: {nn_output[2]}."
        assert nn_output[7] > nn_output[6], f"Start date is after end date. Start: {nn_output[7]}, End: {nn_output[8]}."
        return nn_output
    
