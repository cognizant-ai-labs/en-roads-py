from evolution.candidate import Candidate
from evolution.parent_selection.parent_selector import ParentSelector

class TournamentSelector(ParentSelector):
    """
    Selects parents by doing tournament selection twice.
    """
    def __init__(self, remove_population_pct: float):
        super().__init__(remove_population_pct)
        self.type = "tournament"
    
    def select_parents(self, sorted_parents: list[Candidate], n=2) -> list[Candidate]:
        """
        Selects n parents to mate.
        """
        return [self.tournament_selection(sorted_parents) for _ in range(n)]