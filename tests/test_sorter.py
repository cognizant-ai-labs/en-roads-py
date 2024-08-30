import itertools
import unittest

from evolution.candidate import Candidate
from evolution.sorting.distance_calculation.crowding_distance import CrowdingDistanceCalculator
from evolution.sorting.nsga2_sorter import NSGA2Sorter


class TestSorter(unittest.TestCase):
    def setUp(self):
        crowding_distance = CrowdingDistanceCalculator()
        self.sorter = NSGA2Sorter(crowding_distance)

    def manual_dominates(self, a_asc, b_asc, cand1_a, cand1_b, cand2_a, cand2_b):
        """
        Manual domination of 2 objectives to compare our domination function with.
        """
        better = False
        if a_asc:
            if cand1_a > cand2_a:
                return False
            if cand1_a < cand2_a:
                better = True
        else:
            if cand1_a < cand2_a:
                return False
            if cand1_a > cand2_a:
                better = True
        
        if b_asc:
            if cand1_b > cand2_b:
                return False
            if cand1_b < cand2_b:
                better = True
        else:
            if cand1_b < cand2_b:
                return False
            if cand1_b > cand2_b:
                better = True
        return better


    def test_domination(self):
        """
        Tests domination for all possible combinations of ascension or descending A and B objectives.
        """
        # Iterate over every possible combination of A ascending B descending, etc.
        ascending_combinations = list(itertools.product([True, False], repeat=2))
        for a_asc, b_asc in ascending_combinations:
            cand_config = {"parents": [],
                        "model_params": {"in_size": 1, "hidden_size": 1, "out_size": 1},
                        "actions": ["_source_subsidy_delivered_coal_tce"],
                        "outcomes": {"A": a_asc, "B": b_asc}}

            points = [(0, 0), (1, 1), (0, 1), (1, 0)]
            # Iterate over every possible combination of metric values
            metric_combinations = list(itertools.product(points, repeat=2))
            for (cand1_a, cand1_b), (cand2_a, cand2_b) in metric_combinations:
                candidate1 = Candidate("0_0", **cand_config)
                candidate1.metrics = {"A": cand1_a, "B": cand1_b}
                candidate2 = Candidate("0_1", **cand_config)
                candidate2.metrics = {"A": cand2_a, "B": cand2_b}

                dominates_pred = self.sorter.dominates(candidate1, candidate2)
                dominates_true = self.manual_dominates(a_asc, b_asc, cand1_a, cand1_b, cand2_a, cand2_b)
                self.assertEqual(dominates_pred, dominates_true, msg=f"Failed for {a_asc, b_asc} and {cand1_a, cand1_b} and {cand2_a, cand2_b}")
        
