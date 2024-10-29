"""
Unit Tests for the neural network problem.
"""
import unittest

import torch

from evolution.candidate import Candidate
from moo.problems.nn_problem import candidate_to_params


class TestNNProblem(unittest.TestCase):
    """
    Tests the pymoo neural network problem.
    """
    def test_seed_to_param(self):
        """
        Checks that a candidate model with random parameters can be converted to pymoo parameters and back.
        """
        model_params = {"in_size": 10, "hidden_size": 16, "out_size": 100}

        # Randomly create candidate
        candidate = Candidate("0_0", [], model_params, actions=[])
        for key in candidate.model.state_dict().keys():
            candidate.model.state_dict()[key] = torch.randn(candidate.model.state_dict()[key].shape)

        # Convert to and from pymoo
        params = candidate_to_params(candidate)
        self.assertEqual(len(params), (model_params["in_size"] + 1) * model_params["hidden_size"] +
                         (model_params["hidden_size"] + 1) * model_params["out_size"])
        recons = Candidate.from_pymoo_params(params, model_params, [])

        # Check the models are the same
        for key in candidate.model.state_dict().keys():
            self.assertTrue(torch.equal(candidate.model.state_dict()[key], recons.model.state_dict()[key]))
