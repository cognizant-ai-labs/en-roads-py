"""
Direct representation of actions as a vector of floats from 0 to 1 which are eventually scaled up into the actions.
"""
from pathlib import Path

from presp.prescriptor import Prescriptor, PrescriptorFactory
import torch

from evolution.candidates.output_parser import OutputParser


class DirectPrescriptor(Prescriptor):
    """
    Direct evolution candidate. Simply a vector of floats between 0 and 1 that get rescaled during prescription.
    """
    def __init__(self, actions: list[str]):
        super().__init__()

        self.actions = list(actions)

        self.output_parser = OutputParser(self.actions)
        self.genome = torch.zeros((1, len(actions)), dtype=torch.float32)

    def forward(self, context) -> list[dict]:
        """
        Doesn't require a context tensor input, just converts the genome to a dict of actions by rescaling them.
        """
        with torch.no_grad():
            outputs = self.output_parser.parse_output(self.genome)
            outputs = outputs.cpu().numpy()
        actions_dicts = [dict(zip(self.actions, output.tolist())) for output in outputs]
        return actions_dicts


class DirectFactory(PrescriptorFactory):
    """
    Prescriptor factory handling the construction of direct evolution candidates.
    """
    def __init__(self, actions: list[str]):
        self.actions = list(actions)

    def random_init(self) -> DirectPrescriptor:
        """
        Creates a randomly initialized vector of floats between 0 and 1 uniformly.
        """
        genome = torch.rand((1, len(self.actions)), dtype=torch.float32)
        candidate = DirectPrescriptor(self.actions)
        candidate.genome = genome
        return candidate

    def crossover(self, parents: list[DirectPrescriptor]) -> list[DirectPrescriptor]:
        """
        Crosses over 2 parents using uniform crossover to create a single child.
        """
        parent1 = parents[0].genome
        parent2 = parents[1].genome
        child_genome = torch.where(torch.rand_like(parent1) < 0.5, parent1, parent2)
        child = DirectPrescriptor(self.actions)
        child.genome = child_genome
        return [child]

    def mutation(self, candidate: DirectPrescriptor, mutation_rate: float, mutation_factor: float):
        """
        Mutates the genome of the given candidate in place.
        When the mutation causes a parameter to go out of bounds, mirror it back into bounds.
            For example: mutating 0.9 to 1.2 would cause it to become 0.8
        """
        genome = candidate.genome
        mutate_mask = torch.rand(genome.shape, device=genome.device) < mutation_rate
        noise = torch.normal(0, mutation_factor, genome[mutate_mask].shape, dtype=genome.dtype)
        genome[mutate_mask] *= (1 + noise)

        # Mirror genome back into bounds
        genome = torch.where(genome < 0, -genome, genome)
        genome = torch.where(genome > 1, 2 - genome, genome)

        # Is this necessary?
        candidate.genome = genome

    def save(self, candidate: DirectPrescriptor, path: Path):
        with open(path, "wb") as f:
            # Save the genome as a tensor
            torch.save(candidate.genome, f)

    def load(self, path: Path) -> DirectPrescriptor:
        with open(path, "rb") as f:
            # Load the genome from a tensor
            genome = torch.load(f)
        candidate = DirectPrescriptor(self.actions)
        candidate.genome = genome
        return candidate
