from abc import ABC, abstractmethod

import numpy as np

from evolution.candidate import Candidate

class ParentSelector(ABC):
    """
    Takes a list of sorted parents and selects n parents to mate.
    """
    def __init__(self, remove_population_pct: float, seed: int=None):
        self.remove_pop_pct = remove_population_pct
        self.rng = np.random.default_rng(seed)
    
    @abstractmethod
    def select_parents(self, sorted_parents: list[Candidate], n=2) -> list[Candidate]:
        """
        Selects n parents to mate. Parents should be sorted in descending order.
        """
        raise NotImplementedError

    def tournament_selection(self, sorted_parents: list[Candidate]) -> Candidate:
        """
        Takes 2 random parents and picks the fittest one.
        """
        # Set cutoff to 1 if we are trying to remove more candidates than there are
        cutoff = max(int(len(sorted_parents) * (1 - self.remove_pop_pct)), 1)
        top_parents = sorted_parents[:cutoff]
        return top_parents[min(self.rng.choice(len(top_parents), size=2, replace=True, shuffle=False))]
