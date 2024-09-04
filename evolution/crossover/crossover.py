"""
Abstract crossover class to be inherited.
"""
from abc import ABC, abstractmethod

from evolution.candidate import Candidate
from evolution.mutation.mutation import Mutation


class Crossover(ABC):
    """
    Abstract class for crossover operations.
    """
    def __init__(self, mutator: Mutation = None):
        self.mutator = mutator

    @abstractmethod
    def crossover(self, cand_id: str, parent1: Candidate, parent2: Candidate) -> list[Candidate]:
        """
        Crosses over 2 parents to create offspring. Returns a list so we can return multiple if needed.
        """
        raise NotImplementedError
