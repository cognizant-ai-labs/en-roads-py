"""
Tests for the OutputParser class.
"""
import unittest

import torch

from enroadspy import load_input_specs
from evolution.candidate import OutputParser


class TestOutputParser(unittest.TestCase):
    """
    Tests the output parser class used to parse the output of the neural network into a format usable by enroads.
    """
    def setUp(self):
        self.input_specs = load_input_specs()

    def test_end_values_sliders(self):
        """
        Checks that when we have 0's or 1's the output parser correctly assigns to the min/max indicated in input_specs.
        """
        slider_specs = self.input_specs[self.input_specs["kind"] == "slider"]
        all_slider_actions = list(slider_specs["varId"])

        # Baseline values to compare against
        min_values = list(slider_specs["minValue"])
        max_values = list(slider_specs["maxValue"])

        output_parser = OutputParser(all_slider_actions)

        # Pass all zeroes through output parser, should be min values
        all_zeroes = torch.zeros((1, len(all_slider_actions)))
        zeroes_parsed = output_parser.parse_output(all_zeroes)
        min_values = torch.tensor(min_values, dtype=torch.float32).unsqueeze(0)
        self.assertTrue(torch.isclose(zeroes_parsed, min_values).all())

        # Pass all ones through output parser, should be max values
        all_ones = torch.ones((1, len(all_slider_actions)))
        ones_parsed = output_parser.parse_output(all_ones)
        max_values = torch.tensor(max_values, dtype=torch.float32).unsqueeze(0)
        self.assertTrue(torch.isclose(ones_parsed, max_values).all())

    def test_switch_values(self):
        """
        Checks that when we have a 0/1 for a switch it correctly assigns the on/off value.
        """
        slider_specs = self.input_specs[self.input_specs["kind"] == "switch"]
        all_slider_actions = list(slider_specs["varId"])

        # Baseline values to compare against
        off_values = list(slider_specs["offValue"])
        on_values = list(slider_specs["onValue"])

        output_parser = OutputParser(all_slider_actions)

        # Pass all zeroes through output parser, should be min values
        all_zeroes = torch.zeros((1, len(all_slider_actions)))
        zeroes_parsed = output_parser.parse_output(all_zeroes)
        off_values = torch.tensor(off_values, dtype=torch.float32).unsqueeze(0)
        self.assertTrue(torch.isclose(zeroes_parsed, off_values).all())

        # Pass all ones through output parser, should be max values
        all_ones = torch.ones((1, len(all_slider_actions)))
        ones_parsed = output_parser.parse_output(all_ones)
        on_values = torch.tensor(on_values, dtype=torch.float32).unsqueeze(0)
        self.assertTrue(torch.isclose(ones_parsed, on_values).all())

    def test_all_values(self):
        """
        Comprehensive test over all columns for the above tests.
        """
        # Baseline values to compare against
        slider_specs = self.input_specs.copy()
        all_slider_actions = list(slider_specs["varId"])

        slider_specs["off_merged"] = slider_specs["offValue"].fillna(slider_specs["minValue"])
        slider_specs["on_merged"] = slider_specs["onValue"].fillna(slider_specs["maxValue"])
        
        off_merged = list(slider_specs["off_merged"])
        on_merged = list(slider_specs["on_merged"])

        output_parser = OutputParser(all_slider_actions)

        # Pass all zeroes through output parser, should be min values
        all_zeroes = torch.zeros((1, len(all_slider_actions)))
        zeroes_parsed = output_parser.parse_output(all_zeroes)
        off_merged = torch.tensor(off_merged, dtype=torch.float32).unsqueeze(0)
        self.assertTrue(torch.isclose(zeroes_parsed, off_merged).all())

        # Pass all ones through output parser, should be max values
        all_ones = torch.ones((1, len(all_slider_actions)))
        ones_parsed = output_parser.parse_output(all_ones)
        on_merged = torch.tensor(on_merged, dtype=torch.float32).unsqueeze(0)
        self.assertTrue(torch.isclose(ones_parsed, on_merged).all())

    def test_parse_unparse(self):
        """
        Checks that the parse and unparse functions are inverses of each other.
        NOTE: Qualifying path renewables has the same on and off value so we just ignore it here.
        """
        slider_specs = self.input_specs.copy()
        all_slider_actions = [v for v in list(slider_specs["varId"]) if v != "_qualifying_path_renewables"]
        slider_specs = slider_specs[slider_specs["varId"].isin(all_slider_actions)]

        slider_specs["off_merged"] = slider_specs["offValue"].fillna(slider_specs["minValue"])
        slider_specs["on_merged"] = slider_specs["onValue"].fillna(slider_specs["maxValue"])
        
        off_merged = list(slider_specs["off_merged"])
        on_merged = list(slider_specs["on_merged"])

        output_parser = OutputParser(all_slider_actions)

        # Pass all zeroes through output parser, should be min values
        all_zeroes = torch.zeros((1, len(all_slider_actions)), dtype=torch.float32)
        zeroes_parsed = output_parser.parse_output(all_zeroes)
        off_merged = torch.tensor(off_merged, dtype=torch.float32).unsqueeze(0)
        self.assertTrue(torch.isclose(zeroes_parsed, off_merged).all())

        zeroes_unparsed = output_parser.unparse(zeroes_parsed)
        self.assertTrue(torch.isclose(all_zeroes, zeroes_unparsed).all())

        # Pass all ones through output parser, should be max values
        all_ones = torch.ones((1, len(all_slider_actions)), dtype=torch.float32)
        ones_parsed = output_parser.parse_output(all_ones)
        on_merged = torch.tensor(on_merged, dtype=torch.float32).unsqueeze(0)
        self.assertTrue(torch.isclose(ones_parsed, on_merged).all())

        ones_unparsed = output_parser.unparse(ones_parsed)
        self.assertTrue(torch.isclose(all_ones, ones_unparsed).all())
