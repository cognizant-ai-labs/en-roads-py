"""
Unit tests for the evaluator class.
"""
import unittest

import numpy as np
import pandas as pd
from presp.prescriptor import NNPrescriptorFactory
import torch
import yaml

from enroadspy import load_input_specs, name_to_id
from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.evaluation.evaluator import EnROADSEvaluator
from evolution.utils import process_config


class TestEvaluator(unittest.TestCase):
    """
    Class testing the evaluator class. This is the most important part of the experiment as it executes the
    en-roads model. We need to make sure all our inputs and outputs are correct.
    """
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        # Dummy config to test with
        with open("tests/configs/evaluator.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config = process_config(config)

        self.config = config

        self.evaluator = EnROADSEvaluator(context=config["context"],
                                          actions=config["actions"],
                                          outcomes=config["outcomes"],
                                          n_jobs=config["n_jobs"],
                                          batch_size=config["batch_size"],
                                          device=config["device"])

        self.factory = NNPrescriptorFactory(EnROADSPrescriptor,
                                            config["model_params"],
                                            config["device"],
                                            actions=config["actions"])

    def test_default_input(self):
        """
        Checks that our default input equals the input specs file.
        """
        input_specs = load_input_specs()
        enroads_input = self.evaluator.enroads_runner.construct_enroads_input({})
        split_input = enroads_input.split(" ")
        for i, (default, inp) in enumerate(zip(input_specs["defaultValue"].to_list(), split_input)):
            _, inp_val = inp.split(":")
            self.assertEqual(float(inp_val), float(default), f"Input {i} doesn't match default")

    def test_construct_input(self):
        """
        Tests that only the actions we choose to change are changed in the input.
        """
        input_specs = load_input_specs()
        vals = input_specs["defaultValue"].to_list()

        actions_dict = {"Source tax coal tce": 100}
        enroads_input = self.evaluator.enroads_runner.construct_enroads_input(actions_dict)
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
        input_specs = load_input_specs()
        vals = input_specs["defaultValue"].to_list()

        actions_dict = {"Source tax coal tce": 100}
        enroads_input = self.evaluator.enroads_runner.construct_enroads_input(actions_dict)
        split_input = enroads_input.split(" ")
        for i, (default, inp) in enumerate(zip(vals, split_input)):
            _, inp_val = inp.split(":")
            if i == 0:
                self.assertTrue(np.isclose(float(inp_val), 100), "Didn't modify first input correctly")
            else:
                self.assertTrue(np.isclose(float(inp_val), default), "Messed up first input")

        actions_dict = {"Source tax start time coal": 2040}
        enroads_input = self.evaluator.enroads_runner.construct_enroads_input(actions_dict)
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
        candidate = self.factory.random_init()
        metrics1 = self.evaluator.evaluate_candidate(candidate)
        metrics2 = self.evaluator.evaluate_candidate(candidate)

        self.assertTrue(np.equal(metrics1, metrics2).all())

    def test_checkbox_actions(self):
        """
        Checks to see if the checkboxes actually change the output of the model.
        """
        input_specs = load_input_specs()
        # Switch all checkboxes to true
        true_actions = {}
        for action in self.config["actions"]:
            row = input_specs[input_specs["id"] == action]
            if row["kind"].iloc[0] == "switch":
                true_actions[action] = 1
        true_outcomes = self.evaluator.enroads_runner.evaluate_actions(true_actions)

        # Switch all checkboxes to false
        false_actions = {}
        for action in self.config["actions"]:
            row = input_specs[input_specs["id"] == action]
            if row["kind"].iloc[0] == "switch":
                false_actions[action] = 0
        false_outcomes = self.evaluator.enroads_runner.evaluate_actions(false_actions)

        self.assertFalse(true_outcomes.equals(false_outcomes))

    def test_switches_change_past(self):
        """
        Checks to see if changing each switch messes up the past.
        TODO: We hard-code some exceptions because we believe it's ok for them to change the past.
        """
        input_specs = load_input_specs()
        baseline = self.evaluator.enroads_runner.evaluate_actions({})
        bad_actions = []
        for action in self.config["actions"]:
            row = input_specs[input_specs["id"] == action].iloc[0]
            if row["kind"] == "switch":
                actions_dict = {}
                if row["defaultValue"] == row["onValue"]:
                    actions_dict[action] = row["offValue"]
                else:
                    actions_dict[action] = row["onValue"]
                outcomes = self.evaluator.enroads_runner.evaluate_actions(actions_dict)
                try:
                    pd.testing.assert_frame_equal(outcomes.iloc[:2024-1990], baseline.iloc[:2024-1990])
                except AssertionError:
                    bad_actions.append(action)
        exceptions = ['Apply carbon tax to biofuels',
                      'CCS carbon tax qualifier',
                      'Qualifying path nuclear',
                      'Qualifying path bioenergy',
                      'Qualifying path fossil CCS',
                      'Qualifying path gas']
        exceptions = [name_to_id(action, input_specs) for action in exceptions]
        self.assertEqual(set(bad_actions), set(exceptions), "Switches besides exceptions changed the past")

    def test_sliders_change_past(self):
        """
        Checks to see if setting the slider to the min or max value changes the past.
        """
        input_specs = load_input_specs()
        baseline = self.evaluator.enroads_runner.evaluate_actions({})
        bad_actions = []
        # TODO: When we set this to input_specs['id'].unique() we get some fails we need to account for.
        for action in self.config["actions"]:
            row = input_specs[input_specs["id"] == action].iloc[0]
            if row["kind"] == "slider":
                outcomes_min = self.evaluator.enroads_runner.evaluate_actions({action: row["minValue"]})
                outcomes_max = self.evaluator.enroads_runner.evaluate_actions({action: row["maxValue"]})
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

    def test_start_end_times(self):
        """
        Checks if the end time being before the start time breaks anything.
        """
        actions = ["Source tax coal tce",
                   "Source tax start time coal",
                   "Source tax stop time coal"]
        actions = [name_to_id(action, load_input_specs()) for action in actions]

        input_specs = load_input_specs()
        min_time = input_specs[input_specs["id"] == actions[1]].iloc[0]["minValue"]
        max_time = input_specs[input_specs["id"] == actions[1]].iloc[0]["maxValue"]

        both_start_dict = {actions[0]: -15, actions[1]: min_time, actions[2]: min_time}
        both_end_dict = {actions[0]: -15, actions[1]: max_time, actions[2]: max_time}
        crossed_dict = {actions[0]: -15, actions[1]: max_time, actions[2]: min_time}

        both_start_outcomes = self.evaluator.enroads_runner.evaluate_actions(both_start_dict)
        both_end_outcomes = self.evaluator.enroads_runner.evaluate_actions(both_end_dict)
        crossed_outcomes = self.evaluator.enroads_runner.evaluate_actions(crossed_dict)

        self.assertTrue(both_start_outcomes.equals(both_end_outcomes))
        self.assertTrue(both_start_outcomes.equals(crossed_outcomes))


class TestDecomplexify(unittest.TestCase):
    """
    Tests the decomplexify parameter for the evaluator class.
    """
    def setUp(self):
        with open("tests/configs/decomplexify.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config = process_config(config)
        self.config = config

        self.evaluator = EnROADSEvaluator(context=config["context"],
                                          actions=config["actions"],
                                          outcomes=config["outcomes"],
                                          n_jobs=config["n_jobs"],
                                          batch_size=config["batch_size"],
                                          device=config["device"],
                                          decomplexify=True)

    def test_decomplexify_dict(self):
        """
        Checks that the dict that we expect to pass into the actions when we are decomplexifying is correct.
        """
        true_decomplexify_dict = {
            '_use_subsidies_by_feedstock': 1,
            '_use_new_tech_advanced_settings': 1,
            '_switch_to_use_transport_electrification_detailed_settings': 1,
            '_use_detailed_food_and_ag_controls': 1,
            '_choose_nature_cdr_by_type': 1,
            '_use_detailed_other_ghg_controls': 1,
            '_switch_use_land_detailed_settings': 1,
            '_choose_cdr_by_type': 2
        }

        decomplexify_dict = self.evaluator.decomplexify_dict

        # Check that the keys of the dicts match
        self.assertEqual(set(decomplexify_dict.keys()), set(true_decomplexify_dict.keys()),
                         "Decomplexify dict keys don't match expected keys")

        # Check that the values of the dicts match
        for key in decomplexify_dict:
            self.assertEqual(decomplexify_dict[key], true_decomplexify_dict[key],
                             f"Decomplexify dict {key} doesn't match expected {true_decomplexify_dict[key]}")
