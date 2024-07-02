from evolution.candidate import Candidate
from evolution.sorting.sorter import Sorter
from evolution.sorting.distance_calculation.distance_calculator import DistanceCalculator

class NSGA2Sorter(Sorter):
    def __init__(self, distance_calculator: DistanceCalculator):
        self.distance_calculator = distance_calculator

    def sort_candidates(self, candidates: list[Candidate]):
        # Get ranks of each candidate
        self.fast_non_dominated_sort(candidates)
        self.distance_calculator.calculate_distance(candidates)

        # Sort primarily by rank, secondarily by distance
        candidates.sort(key=lambda x: (x.rank, -x.distance))
        return candidates

    # pylint: disable=consider-using-enumerate
    def fast_non_dominated_sort(self, candidates: list[Candidate]):
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
        """
        better = False
        for obj in candidate1.metrics.keys():
            if candidate1.metrics[obj] < candidate2.metrics[obj]:
                return False
            if candidate1.metrics[obj] > candidate2.metrics[obj]:
                better = True
        return better