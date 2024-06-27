from abc import ABC, abstractmethod

from evolution.candidate import Candidate

class DistanceCalculator(ABC):
    """
    Class to calculate distances between candidates
    """
    @abstractmethod
    def calculate_distance(self, front: list[Candidate]) -> None:
        """
        Calculates the distances of each candidate in the front.
        """
        raise NotImplementedError

