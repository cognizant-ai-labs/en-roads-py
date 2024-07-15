"""
Unit tests for the evaluator class.
"""
import shutil
import unittest

import numpy as np
import pandas as pd

from evolution.utils import modify_config
from evolution.candidate import Candidate
from evolution.evaluation.evaluator import Evaluator

class TestEvaluator(unittest.TestCase):
    """
    Class testing the evaluator class. This is the most important part of the experiment as it executes the
    en-roads model. We need to make sure all our inputs and outputs are correct.
    """
    def setUp(self):
        # Dummy config to test with
        config = {
            "evolution_params": {
                "n_generations": 100,
                "pop_size": 100,
                "n_elites": 10
            },
            "remove_population_pct": 0.7,
            "mutation_factor": 0.1,
            "mutation_rate": 0.1,
            "model_params": {
                "hidden_size": 64
            },
            "eval_params": {
                "temp_dir": "tests/temp"
            },
            "context": [
                "_new_tech_breakthrough_setting"
            ],
            "actions": [
                "_source_subsidy_delivered_coal_tce",
                "_source_subsidy_start_time_delivered_coal",
                "_source_subsidy_stop_time_delivered_coal",
                "_no_new_coal",
                "_year_of_no_new_capacity_coal",
                "_utilization_adjustment_factor_delivered_coal",
                "_utilization_policy_start_time_delivered_coal",
                "_utilization_policy_stop_time_delivered_coal",
                "_target_accelerated_retirement_rate_electric_coal",
                "_source_subsidy_delivered_oil_boe",
                "_source_subsidy_start_time_delivered_oil",
                "_source_subsidy_stop_time_delivered_oil",
                "_no_new_oil",
                "_year_of_no_new_capacity_oil",
                "_utilization_adjustment_factor_delivered_oil",
                "_utilization_policy_start_time_delivered_oil",
                "_utilization_policy_stop_time_delivered_oil",
                "_source_subsidy_delivered_gas_mcf",
                "_source_subsidy_start_time_delivered_gas",
                "_source_subsidy_stop_time_delivered_gas",
                "_no_new_gas",
                "_year_of_no_new_capacity_gas",
                "_utilization_adjustment_factor_delivered_gas",
                "_utilization_policy_start_time_delivered_gas",
                "_utilization_policy_stop_time_delivered_gas",
                "_source_subsidy_renewables_kwh",
                "_source_subsidy_start_time_renewables",
                "_source_subsidy_stop_time_renewables",
                "_use_subsidies_by_feedstock",
                "_source_subsidy_delivered_bio_boe",
                "_source_subsidy_start_time_delivered_bio",
                "_source_subsidy_stop_time_delivered_bio",
                "_no_new_bio",
                "_year_of_no_new_capacity_bio",
                "_wood_feedstock_subsidy_boe",
                "_crop_feedstock_subsidy_boe",
                "_other_feedstock_subsidy_boe",
                "_source_subsidy_nuclear_kwh",
                "_source_subsidy_start_time_nuclear",
                "_source_subsidy_stop_time_nuclear",
                "_carbon_tax_initial_target",
                "_carbon_tax_phase_1_start",
                "_carbon_tax_time_to_achieve_initial_target",
                "_carbon_tax_final_target",
                "_carbon_tax_phase_3_start",
                "_carbon_tax_time_to_achieve_final_target",
                "_apply_carbon_tax_to_biofuels",
                "_ccs_carbon_tax_qualifier",
                "_qualifying_path_renewables",
                "_qualifying_path_nuclear",
                "_qualifying_path_new_zero_carbon",
                "_qualifying_path_beccs",
                "_qualifying_path_bioenergy",
                "_qualifying_path_fossil_ccs",
                "_qualifying_path_gas",
                "_electric_standard_active",
                "_electric_standard_target",
                "_electric_standard_start_year",
                "_electric_standard_target_time",
                "_emissions_performance_standard",
                "_performance_standard_time"
            ],
            "outcomes": [
                "Net cumulative emissions",
                "Total cost of energy",
                "Cost of energy next 10 years"
            ],
            "save_path": "tests/blah"
        }
        config = modify_config(config)
        self.config = config
        self.evaluator = Evaluator(**self.config["eval_params"])

    def test_default_input(self):
        """
        Checks that our default input equals the input specs file.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        self.evaluator.construct_enroads_input({})
        with open("tests/temp/enroads_input.txt", "r", encoding="utf-8") as f:
            enroads_input = f.read()
            split_input = enroads_input.split(" ")
            for i, (default, inp) in enumerate(zip(input_specs["defaultValue"].to_list(), split_input)):
                _, inp_val = inp.split(":")
                self.assertEqual(float(inp_val), float(default), f"Input {i} doesn't match default")

    def test_construct_input(self):
        """
        Tests that only the actions we choose to change are changed in the input.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        vals = input_specs["defaultValue"].to_list()

        self.evaluator.construct_enroads_input({"_source_subsidy_delivered_coal_tce": 100})
        with open("tests/temp/enroads_input.txt", "r", encoding="utf-8") as f:
            enroads_input = f.read()
            split_input = enroads_input.split(" ")
            for i, (default, inp) in enumerate(zip(vals, split_input)):
                _, inp_val = inp.split(":")
                if i == 0:
                    self.assertEqual(float(inp_val), 100, "Didn't modify first input correctly")
                else:
                    self.assertEqual(float(inp_val), default, f"Messed up input {i}: {inp_val} != {default}")

    def test_repeat_constructs(self):
        """
        Tests inputs don't contaminate each other.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        vals = input_specs["defaultValue"].to_list()

        self.evaluator.construct_enroads_input({"_source_subsidy_delivered_coal_tce": 100})
        with open("tests/temp/enroads_input.txt", "r", encoding="utf-8") as f:
            enroads_input = f.read()
            split_input = enroads_input.split(" ")
            for i, (default, inp) in enumerate(zip(vals, split_input)):
                _, inp_val = inp.split(":")
                if i == 0:
                    self.assertTrue(np.isclose(float(inp_val), 100), "Didn't modify first input correctly")
                else:
                    self.assertTrue(np.isclose(float(inp_val), default), "Messed up first input")

        self.evaluator.construct_enroads_input({"_source_subsidy_start_time_delivered_coal": 2040})
        with open("tests/temp/enroads_input.txt", "r", encoding="utf-8") as f:
            enroads_input = f.read()
            split_input = enroads_input.split(" ")
            for i, (default, inp) in enumerate(zip(vals, split_input)):
                _, inp_val = inp.split(":")
                if i == 1:
                    self.assertTrue(np.isclose(float(inp_val), 2040), "Didn't modify second input correctly")
                else:
                    self.assertTrue(np.isclose(float(inp_val), default), "Second input contaminated")

    def test_consistent_eval(self):
        """
        Makes sure that the same candidate evaluated twice has the same metrics.
        """
        candidate = Candidate("0_0", [], self.config["model_params"], self.config["actions"], self.config["outcomes"])
        self.evaluator.evaluate_candidate(candidate)
        original = {k: v for k, v in candidate.metrics.items()}

        self.evaluator.evaluate_candidate(candidate)

        for outcome in self.config["outcomes"]:
            self.assertEqual(original[outcome], candidate.metrics[outcome])

    def test_checkbox_actions(self):
        """
        Checks to see if the checkboxes actually change the output of the model.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        # Switch all checkboxes to true
        true_actions = {}
        for action in self.config["actions"]:
            row = input_specs[input_specs["varId"] == action]
            if row["kind"].iloc[0] == "switch":
                true_actions[action] = 1
        true_outcomes = self.evaluator.evaluate_actions(true_actions)

        # Switch all checkboxes to false
        false_actions = {}
        for action in self.config["actions"]:
            row = input_specs[input_specs["varId"] == action]
            if row["kind"].iloc[0] == "switch":
                false_actions[action] = 0
        false_outcomes = self.evaluator.evaluate_actions(false_actions)

        self.assertFalse(true_outcomes.equals(false_outcomes))

    def test_switches_change_past(self):
        """
        Checks to see if changing each switch messes up the past.
        TODO: This test is failing because of a bug in en-roads?
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        baseline = self.evaluator.evaluate_actions({})
        bad_actions = []
        for action in self.config["actions"]:
            row = input_specs[input_specs["varId"] == action].iloc[0]
            if row["kind"] == "switch":
                actions_dict = {}
                if row["defaultValue"] == row["onValue"]:
                    actions_dict[action] = row["offValue"]
                else:
                    actions_dict[action] = row["onValue"]
                outcomes = self.evaluator.evaluate_actions(actions_dict)
                try:
                    pd.testing.assert_frame_equal(outcomes.iloc[:2024-1990], baseline.iloc[:2024-1990])
                except AssertionError:
                    bad_actions.append(action)
        self.assertEqual(len(bad_actions), 0, f"Switches {bad_actions} changed the past")

    def test_sliders_change_past(self):
        """
        Checks to see if setting the slider to the min or max value changes the past.
        """
        input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)
        baseline = self.evaluator.evaluate_actions({})
        bad_actions = []
        # TODO: When we set this to input_specs['varId'].unique() we get some fails we need to account for.
        for action in self.config["actions"]:
            row = input_specs[input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                outcomes_min = self.evaluator.evaluate_actions({action: row["minValue"]})
                outcomes_max = self.evaluator.evaluate_actions({action: row["maxValue"]})
                try:
                    pd.testing.assert_frame_equal(outcomes_min.iloc[:2024-1990],
                                                  baseline.iloc[:2024-1990],
                                                  check_dtype=False)
                    pd.testing.assert_frame_equal(outcomes_max.iloc[:2024-1990],
                                                  baseline.iloc[:2024-1990],
                                                  check_dtype=False)
                except AssertionError:
                    bad_actions.append(action)

        self.assertEqual(len(bad_actions), 0, f"Sliders {bad_actions} changed the past")

    def tearDown(self):
        shutil.rmtree("tests/temp")

