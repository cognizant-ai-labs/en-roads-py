"""
Unit tests for the pymoo evolution process.
"""
import json
from pathlib import Path
import shutil
import unittest

import dill

from moo.run_pymoo import optimize


class TestPymoo(unittest.TestCase):
    """
    Tests that we can run through the pymoo evolution process and nothing errors.
    This is mostly just to check for runtime errors rather than behavior.
    """

    def setUp(self):
        shutil.rmtree("tests/temp")
        with open("tests/configs/dummy.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
        Path(self.config["save_path"]).mkdir(parents=True)

    def test_generic_evolution_default(self):
        """
        Runs 2 generations of evolution with the default problem tests that the results and candidates are the right
        size.
        Results: n outcomes
        Candidates: n actions
        """
        optimize(self.config, False)

        with open("tests/temp/dummy/results", "rb") as f:
            res = dill.load(f)

        self.assertEqual(res.F.shape[1], len(self.config["outcomes"]))
        self.assertEqual(res.X.shape[1], len(self.config["actions"]))

    def test_generic_evolution_nn(self):
        """
        Runs 2 generations of evolution with the default problem tests that the results and candidates are the right
        shape.
        Results: n outcomes
        Candidates: n params
        """
        optimize(self.config, True)

        with open("tests/temp/dummy/results", "rb") as f:
            res = dill.load(f)

        self.assertEqual(res.F.shape[1], len(self.config["outcomes"]))

        in_size = res.problem.model_params[0]["in_features"]
        hidden_size = res.problem.model_params[0]["out_features"]
        out_size = res.problem.model_params[2]["out_features"]
        num_params = (in_size + 1) * hidden_size + (hidden_size + 1) * out_size
        self.assertEqual(res.X.shape[1], num_params)

    def tearDown(self):
        shutil.rmtree("tests/temp")
