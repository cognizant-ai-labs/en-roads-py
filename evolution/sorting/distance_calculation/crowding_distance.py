from evolution.candidate import Candidate
from evolution.sorting.distance_calculation.distance_calculator import DistanceCalculator

class CrowdingDistanceCalculator(DistanceCalculator):
    """
    Calculates NSGA-II crowding distance
    """
    def __init__(self):
        self.type = "crowding"

    def calculate_distance(self, front: list[Candidate]) -> None:
        """
        Calculate crowding distance of each candidate in front and set it as the distance attribute.
        Candidates are assumed to already have metrics computed.
        """
        for c in front:
            c.distance = 0
        
        # Front is sorted by each metric
        for m in front[0].metrics.keys():
            front.sort(key=lambda c: c.metrics[m])

            # If a candidate has a bad metric, set its distance to negative infinity and ignore it
            start = 0
            while front[start].metrics[m] == float("-inf") or front[start].metrics[m] == float("inf"):
                front[start].distance = float('-inf')
                start += 1
            end = -1
            while front[end].metrics[m] == float("-inf") or front[end].metrics[m] == float("inf"):
                front[end].distance = float('-inf')
                end -= 1

            # Standard NSGA-II Crowding Distance calculation
            obj_min = front[start].metrics[m]
            obj_max = front[end].metrics[m]
            front[start].distance = float('inf')
            front[end].distance = float('inf')
            for i in range(start + 1, len(front) + end - 1):
                # We hard-code simulator fails as inf and have to do this to avoid warnings
                assert front[i].metrics[m] != float("-inf") and front[i].metrics[m] != float("inf"), "Metric should not be -inf or inf"
                if obj_max != obj_min:
                    front[i].distance += (front[i+1].metrics[m] - front[i-1].metrics[m]) / (obj_max - obj_min)
                # If all candidates have the same value, their distances are 0
                else:
                    front[i].distance += 0
