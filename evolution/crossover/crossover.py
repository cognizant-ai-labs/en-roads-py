from abc import ABC, abstractmethod
from pathlib import Path

from evolution.candidate import Candidate
from evolution.mutation.mutation import Mutation

class Crossover(ABC):
    """
    Abstract class for crossover operations.
    """
    def __init__(self, full=False, mutator: Mutation=None):
        self.mutator = mutator
    
    @abstractmethod
    def crossover(self, cand_id: str, parent1: Candidate, parent2: Candidate) -> list[Candidate]:
        raise NotImplementedError