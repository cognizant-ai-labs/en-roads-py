import shutil
import unittest

import numpy as np
import pandas as pd

from evolution.utils import modify_config
from evolution.candidate import Candidate
from evolution.evaluation.evaluator import Evaluator

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        config = {
            "model_params": {
                "hidden_size": 16
            },
            "eval_params": {
                "temp_dir": "tests/temp",
            },
            "context": [],
            "actions": [
                "_source_subsidy_delivered_coal_tce",
                "_source_subsidy_start_time_delivered_coal",
                "_source_subsidy_stop_time_delivered_coal",
                "_no_new_coal",
                "_year_of_no_new_capacity_coal",
                "_utilization_adjustment_factor_delivered_coal",
                "_utilization_policy_start_time_delivered_coal",
                "_utilization_policy_stop_time_delivered_coal",
                "_target_accelerated_retirement_rate_electric_coal"
            ],
            "outcomes": [
                "CO2 Gross emissions",
                "Total cost of energy"
            ]
        }
        config = modify_config(config)
        self.config = config

    def test_construct_input(self):
        """
        Tests that only the actions we choose to change are changed in the input.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True)
        vals = input_specs["defaultValue"].to_list()

        evaluator = Evaluator(**self.config["eval_params"])
        evaluator.construct_enroads_input({"_source_subsidy_delivered_coal_tce": 100})
        with open("tests/temp/enroads_input.txt", "r", encoding="utf-8") as f:
            enroads_input = f.read()
            split_input = enroads_input.split(" ")
            for i, (default, inp) in enumerate(zip(vals, split_input)):
                _, inp_val = inp.split(":")
                if i == 0:
                    self.assertTrue(np.isclose(float(inp_val), 100))
                else:
                    self.assertTrue(np.isclose(float(inp_val), default))

    def test_repeat_constructs(self):
        """
        Tests inputs don't contaminate each other.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True)
        vals = input_specs["defaultValue"].to_list()

        evaluator = Evaluator(**self.config["eval_params"])
        evaluator.construct_enroads_input({"_source_subsidy_delivered_coal_tce": 100})
        with open("tests/temp/enroads_input.txt", "r", encoding="utf-8") as f:
            enroads_input = f.read()
            split_input = enroads_input.split(" ")
            for i, (default, inp) in enumerate(zip(vals, split_input)):
                _, inp_val = inp.split(":")
                if i == 0:
                    self.assertTrue(np.isclose(float(inp_val), 100))
                else:
                    self.assertTrue(np.isclose(float(inp_val), default))

        evaluator.construct_enroads_input({"_source_subsidy_start_time_delivered_coal": 2040})
        with open("tests/temp/enroads_input.txt", "r", encoding="utf-8") as f:
            enroads_input = f.read()
            split_input = enroads_input.split(" ")
            for i, (default, inp) in enumerate(zip(vals, split_input)):
                _, inp_val = inp.split(":")
                if i == 1:
                    self.assertTrue(np.isclose(float(inp_val), 2040))
                else:
                    self.assertTrue(np.isclose(float(inp_val), default))

    def test_consistent_eval(self):
        """
        Makes sure that the same candidate evaluated twice has the same metrics.
        """
        evaluator = Evaluator(**self.config["eval_params"])
        candidate = Candidate("0_0", [], self.config["model_params"], self.config["actions"], self.config["outcomes"])
        evaluator.evaluate_candidate(candidate)
        original = {k: v for k, v in candidate.metrics.items()}

        evaluator.evaluate_candidate(candidate)

        for outcome in self.config["outcomes"]:
            self.assertEqual(original[outcome], candidate.metrics[outcome])


    def tearDown(self):
        shutil.rmtree("tests/temp")

