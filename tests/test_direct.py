"""
Unittests for the DirectPrescriptor class which is a direct representation of actions as a vector of floats from 0 to 1.
"""
import unittest

import torch
import yaml

from enroadspy import load_input_specs, BAD_SWITCH
from evolution.candidates.direct import DirectPrescriptor, DirectFactory
from evolution.candidates.output_parser import OutputParser
from evolution.seeding.train_seeds import create_direct_seeds


class TestDirect(unittest.TestCase):
    """
    Tests the DirectPrescriptor class.
    """
    def setUp(self):
        with open("tests/configs/direct.yml", "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.actions = list(self.config["actions"])
        self.output_parser = OutputParser(self.config["actions"])
        self.factory = DirectFactory(self.actions)

        self.input_specs = load_input_specs()

    def check_candidate(self, candidate: DirectPrescriptor, cand_type="default"):
        """
        Checks a candidate to make sure it's outputting the correct values.
        If cand_type is default its output is compared to the default values. This goes for "min" and "max" as well.
        """
        actions_dict = candidate.forward(None)[0]

        for action in actions_dict:
            row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                truth = None
                if cand_type == "default":
                    truth = row["defaultValue"]
                elif cand_type == "min":
                    truth = row["minValue"]
                elif cand_type == "max":
                    truth = row["maxValue"]
                else:
                    raise ValueError(f"Unknown candidate type {cand_type}")

                self.assertAlmostEqual(actions_dict[action], truth, places=5)
                # self.assertTrue(np.isclose(actions_dict[action], truth))
            else:
                truth = None
                if cand_type == "default":
                    truth = row["defaultValue"]
                elif cand_type == "min":
                    truth = row["offValue"]
                elif cand_type == "max":
                    truth = row["onValue"]
                else:
                    raise ValueError(f"Unknown candidate type {cand_type}")
                self.assertAlmostEqual(actions_dict[action], truth, places=5)

    def test_uninitialized(self):
        """
        Tests to make sure that an uninitialized DirectPrescriptor returns all zeros and all these zeroes are
        translated to the correct values.
        """
        uninitialized = DirectPrescriptor(self.actions)
        genome = uninitialized.genome

        self.assertTrue(torch.equal(genome, torch.zeros((1, len(self.actions)), dtype=torch.float32)))

        self.check_candidate(uninitialized, "min")

    def test_seeding(self):
        """
        Checks that the min, max, and default seeds are initialized correctly and output the correct values when
        forward() is called.
        """
        direct_seeds = create_direct_seeds(self.actions)

        # Check the seeds output the correct value when forward() is called
        min_seed = direct_seeds[0]
        max_seed = direct_seeds[1]
        default_seed = direct_seeds[2]
        self.check_candidate(min_seed, "min")
        self.check_candidate(max_seed, "max")
        self.check_candidate(default_seed, "default")

        # Check that the genomes are the correct scaled values
        min_genome, max_genome, default_genome = min_seed.genome, max_seed.genome, default_seed.genome

        # NOTE: The bad switch has the same on and off value so we manually set it to 0 to pass the test.
        bad_idx = self.actions.index(BAD_SWITCH)
        min_genome[0, bad_idx] = 0

        self.assertTrue(torch.allclose(min_genome, torch.zeros((1, len(self.actions)), dtype=torch.float32)))
        self.assertTrue(torch.allclose(max_genome, torch.ones((1, len(self.actions)), dtype=torch.float32)))

        # Get the true values of the default genome
        default_values = {}
        for action in self.actions:
            row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
            default_values[action] = row["defaultValue"]
        unscaled_default = torch.tensor(list(default_values.values()), dtype=torch.float32).unsqueeze(0)
        default_tensor = self.output_parser.unparse(unscaled_default)

        self.assertTrue(torch.allclose(default_genome, default_tensor))
