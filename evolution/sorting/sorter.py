from abc import ABC, abstractmethod

from evolution.candidate import Candidate

class Sorter(ABC):
    """
    Abstract class that handles the sorting of candidates after they are evaluated.
    """

    @abstractmethod
    def sort_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        """
        Sorts candidates based on some criteria.
        """
        raise NotImplementedError
