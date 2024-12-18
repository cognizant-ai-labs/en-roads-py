import copy
from pathlib import Path

from presp.prescriptor import NNPrescriptor, PrescriptorFactory
import torch

from evolution.candidate import OutputParser


class EnROADSPrescriptor(NNPrescriptor):
    def __init__(self, actions: list[str], model_params: dict, device: str = "cpu"):
        super().__init__(model_params, device)
        self.actions = actions
        self.output_parser = OutputParser(actions, device)

    def forward(self, context):
        temp = torch.zeros(1, self.model_params["in_size"], device=self.device)
        output = self.model(temp)
        parsed = self.output_parser.parse_output(output)[0]
        actions_dict = dict(zip(self.actions, parsed.tolist()))
        return actions_dict

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)


class EnROADSPrescriptorFactory(PrescriptorFactory):
    def __init__(self, actions: list[str], model_params: dict, device: str = "cpu"):
        self.model_params = model_params
        self.device = device
        self.actions = actions

    def random_init(self) -> EnROADSPrescriptor:
        prescriptor = EnROADSPrescriptor(self.actions, self.model_params, self.device)
        # TODO: We can't do orthogonal init on mps so go to cpu then back
        prescriptor.model.to("cpu")
        for layer in prescriptor.model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                layer.bias.data.fill_(0.01)
        prescriptor.model.to(self.device)
        return prescriptor

    def crossover(self, parents: list[EnROADSPrescriptor], mutation_rate: float, mutation_factor: float) -> list[EnROADSPrescriptor]:
        child = EnROADSPrescriptor(self.actions, self.model_params, self.device)
        parent1, parent2 = parents[0], parents[1]
        child.model = copy.deepcopy(parent1.model)
        for child_param, parent2_param in zip(child.model.parameters(), parent2.model.parameters()):
            mask = torch.rand(size=child_param.data.shape, device=self.device) < 0.5
            child_param.data[mask] = parent2_param.data[mask]
        self.mutate_(child, mutation_rate, mutation_factor)
        return [child]

    def mutate_(self, candidate: EnROADSPrescriptor, mutation_rate: float, mutation_factor: float):
        """
        Mutates a prescriptor in-place with gaussian percent noise.
        """
        with torch.no_grad():
            for param in candidate.model.parameters():
                mutate_mask = torch.rand(param.shape, device=param.device) < mutation_rate
                noise = torch.normal(0, mutation_factor, param[mutate_mask].shape, device=param.device, dtype=param.dtype)
                param[mutate_mask] += noise * param[mutate_mask]

    def load(self, path: Path) -> EnROADSPrescriptor:
        prescriptor = EnROADSPrescriptor(self.actions, self.model_params, self.device)
        prescriptor.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return prescriptor
