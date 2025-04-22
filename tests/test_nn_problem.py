"""
Unit Tests for the neural network problem.
"""
import unittest

from presp.prescriptor import NNPrescriptorFactory
import torch

from evolution.candidates.candidate import EnROADSPrescriptor
from moo.problems.nn_problem import candidate_to_params


class TestNNProblem(unittest.TestCase):
    """
    Tests the pymoo neural network problem.
    """
    def test_seed_to_param(self):
        """
        Checks that a candidate model with random parameters can be converted to pymoo parameters and back.
        """
        model_params = [
            {"type": "linear", "in_features": 10, "out_features": 16},
            {"type": "tanh"},
            {"type": "linear", "in_features": 16, "out_features": 100},
            {"type": "sigmoid"}
        ]
        device = "cpu"
        factory = NNPrescriptorFactory(EnROADSPrescriptor, model_params, device, actions=[])

        # Randomly create candidate
        candidate = factory.random_init()
        for key in candidate.model.state_dict().keys():
            candidate.model.state_dict()[key] = torch.randn(candidate.model.state_dict()[key].shape)

        # Convert to and from pymoo
        params = candidate_to_params(candidate)

        # Check the number of parameters is correct
        n_params = 0
        for layer in model_params:
            if layer["type"] == "linear":
                n_params += (layer["in_features"] + 1) * layer["out_features"]
        self.assertEqual(len(params), n_params)

        # Reconstruct candidate from params
        recons = EnROADSPrescriptor.from_pymoo_params(params, model_params, actions=[])

        # Check the models are the same
        for key in candidate.model.state_dict().keys():
            self.assertTrue(torch.equal(candidate.model.state_dict()[key], recons.model.state_dict()[key]))
