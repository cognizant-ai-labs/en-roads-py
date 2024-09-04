"""
Sorter implementation sorting via. NSGA-II.
"""
from evolution.candidate import Candidate
from evolution.sorting.sorter import Sorter
from evolution.sorting.distance_calculation.distance_calculator import DistanceCalculator


class NSGA2Sorter(Sorter):
    """
    Sorts candidates using NSGA-II.
    Performs non-dominated sort to get the ranks of each candidate then within each front calculates distance and sorts
    by them.
    """
    def __init__(self, distance_calculator: DistanceCalculator):
        self.distance_calculator = distance_calculator

    def sort_candidates(self, candidates: list[Candidate]):
        """
        Computes rank and distance within each front, then sorts by front then distance ascending and descending.
        """
        # Get ranks of each candidate
        fronts = self.fast_non_dominated_sort(candidates)

        # Calculate distance for each front
        for front in fronts:
            self.distance_calculator.calculate_distance(front)

        # Sort primarily by rank, secondarily by distance
        candidates.sort(key=lambda x: (x.rank, -x.distance))
        return candidates

    # pylint: disable=consider-using-enumerate
    def fast_non_dominated_sort(self, candidates: list[Candidate]):
        """
        NSGA-II fast non dominated sort algorithm.
        """
        S = [[] for _ in range(len(candidates))]
        n = [0 for _ in range(len(candidates))]
        fronts = [[]]
        for p in range(len(candidates)):
            S[p] = []
            n[p] = 0
            for q in range(len(candidates)):
                if self.dominates(candidates[p], candidates[q]):
                    S[p].append(q)
                elif self.dominates(candidates[q], candidates[p]):
                    n[p] += 1
            if n[p] == 0:
                candidates[p].rank = 1
                fronts[0].append(p)

            # print(f"S[{p}]: {S[p]}")
            # print(f"n[{p}]: {n[p]}")

        i = 1
        while len(fronts[i-1]) > 0:
            Q = []
            for p in fronts[i-1]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        candidates[q].rank = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)

        cand_fronts = []
        for front in fronts:
            if len(front) > 0:
                cand_fronts.append([candidates[i] for i in front])
        return cand_fronts

    # pylint: enable=consider-using-enumerate

    def dominates(self, candidate1: Candidate, candidate2: Candidate) -> bool:
        """
        Determine if one individual dominates another.
        If an objective is ascending we want to minimize it.
            If cand2 < cand1 for at least one objective we return False.
            If cand1 < cand2 for at least one objective and not the above we return True.
            If cand1 == cand2 for all objectives we return False.
        If an objective is not ascending we want to maximize it.
            If cand2 > cand1 for at least one objective we return False.
            If cand1 > cand2 for at least one objective and not the above we return True.
            If cand1 == cand2 for all objectives we return False
        """
        assert candidate1.outcomes.keys() == candidate2.outcomes.keys(), \
            "Candidates must have the same objectives to compare them."
        better = False
        for obj, ascending in candidate1.outcomes.items():
            if ascending:
                if candidate1.metrics[obj] > candidate2.metrics[obj]:
                    return False
                if candidate1.metrics[obj] < candidate2.metrics[obj]:
                    better = True
            else:
                if candidate1.metrics[obj] < candidate2.metrics[obj]:
                    return False
                if candidate1.metrics[obj] > candidate2.metrics[obj]:
                    better = True
        return better
