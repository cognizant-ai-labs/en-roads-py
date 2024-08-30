import unittest

import pandas as pd

from generate_url import actions_to_url, generate_actions_dict
import webbrowser

class TestURLGenerator(unittest.TestCase):

    def setUp(self):
        self.input_specs = pd.read_json("inputSpecs.jsonl", lines=True, precise_float=True)

    def test_generate_url(self):
        """
        Tests if we change every single action to a different value, we can construct a URL then deconstruct it
        back into the same dictionary.
        """
        actions_dict = {}
        for action in self.input_specs["varId"].tolist():
            row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                actions_dict[action] = row["maxValue"] if row["defaultValue"] != row["maxValue"] else row["minValue"]
            else:
                actions_dict[action] = row["onValue"] if row["defaultValue"] != row["onValue"] else row["offValue"]
        
        url = actions_to_url(actions_dict)
        reverse_url = generate_actions_dict(url)

        self.assertEqual(actions_dict, reverse_url)

    # def test_default_url(self):
    #     """
    #     Tests that the default URL sends us to a page with no actions.
    #     """
    #     actions_dict = {}
    #     for action in self.input_specs["varId"].tolist():
    #         row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
    #         actions_dict[action] = int(row["defaultValue"])
    #         url = actions_to_url(actions_dict)
    #         webbrowser.open(url)
    #         actions_dict.pop(action)

    #     self.assertEqual(False, True)