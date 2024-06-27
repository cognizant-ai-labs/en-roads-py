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
        for m in front[0].metrics.keys():
            front.sort(key=lambda c: c.metrics[m])
            obj_min = front[0].metrics[m]
            obj_max = front[-1].metrics[m]
            front[0].distance = float('inf')
            front[-1].distance = float('inf')
            for i in range(1, len(front) - 1):
                if obj_max != obj_min:
                    front[i].distance += (front[i+1].metrics[m] - front[i-1].metrics[m]) / (obj_max - obj_min)
                # If all candidates have the same value, their distances are 0
                else:
                    front[i].distance += 0