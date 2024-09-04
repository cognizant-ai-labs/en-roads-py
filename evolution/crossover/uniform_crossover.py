"""
Standard uniform crossover
"""
import torch

from evolution.candidate import Candidate
from evolution.crossover.crossover import Crossover
from evolution.mutation.mutation import Mutation


class UniformCrossover(Crossover):
    """
    Crosses over 2 parents.
    We do not keep track of what's in the models and assume they are loaded correct with the parents.
    """
    def __init__(self, mutator: Mutation = None):
        super().__init__(mutator)
        self.type = "uniform"

    def crossover(self, cand_id: str, parent1: Candidate, parent2: Candidate) -> list[Candidate]:
        child = Candidate(cand_id,
                          [parent1.cand_id, parent2.cand_id],
                          parent1.model_params,
                          parent1.actions,
                          parent1.outcomes)
        with torch.no_grad():
            child.model.load_state_dict(parent1.model.state_dict())
            for param, param2 in zip(child.model.parameters(), parent2.model.parameters()):
                mask = torch.rand(param.shape, device=param.device) < 0.5
                param.data[mask] = param2.data[mask]
        self.mutator.mutate_(child)
        return [child]
