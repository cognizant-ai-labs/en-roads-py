"""
Tests URL Generation.
"""
import unittest

from enroadspy import load_input_specs
from enroadspy.generate_url import actions_to_url, generate_actions_dict


class TestURLGenerator(unittest.TestCase):
    """
    Class to test URL generation converting context/actions to an En-ROADS URL.
    TODO: The default URL doesn't work for some reason.
    """
    def setUp(self):
        self.input_specs = load_input_specs()

    def test_generate_url(self):
        """
        Tests if we change every single action to a different value, we can construct a URL then deconstruct it
        back into the same dictionary.
        """
        actions_dict = {}
        for action in self.input_specs["id"].tolist():
            row = self.input_specs[self.input_specs["id"] == action].iloc[0]
            if row["kind"] == "slider":
                actions_dict[action] = row["maxValue"] if row["defaultValue"] != row["maxValue"] else row["minValue"]
            else:
                actions_dict[action] = row["onValue"] if row["defaultValue"] != row["onValue"] else row["offValue"]

        url = actions_to_url(actions_dict)
        reverse_url = generate_actions_dict(url)

        self.assertEqual(actions_dict, reverse_url)
