"""
Unittest for distance calculation for NSGA-II crowding distance.
"""
import unittest

from evolution.candidate import Candidate
from evolution.sorting.distance_calculation.crowding_distance import CrowdingDistanceCalculator


class TestDistanceCalculation(unittest.TestCase):
    """
    Class for testing the crowding distance calculation.
    """
    def setUp(self):
        self.distance_calculator = CrowdingDistanceCalculator()

    def test_crowding_distance_inf(self):
        """
        Creates dummy candidates and tests the crowding distance calculation on some synthetic metrics.
        All the candidates are equal according to the pareto so they should all have a distance of infinity.
        """
        cand_params = {
            "parents": [],
            "model_params": {"in_size": 1, "hidden_size": 1, "out_size": 1},
            "actions": ["_source_subsidy_delivered_coal_tce"],
        }
        cand1 = Candidate("0_0", **cand_params)
        cand2 = Candidate("0_1", **cand_params)
        cand3 = Candidate("0_2", **cand_params)
        cand4 = Candidate("0_3", **cand_params)

        cand1.metrics = {"A": 10, "B": 1}
        cand2.metrics = {"A": 1, "B": 10}
        cand3.metrics = {"A": 0, "B": 1}
        cand4.metrics = {"A": 1, "B": 0}

        candidates = [cand1, cand2, cand3, cand4]
        self.distance_calculator.calculate_distance(candidates)
        for cand in candidates:
            self.assertEqual(cand.distance, float('inf'))
