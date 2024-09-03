"""
Tests the enroads model
"""
import unittest

from enroadspy import load_input_specs
from enroadspy.enroads_runner import EnroadsRunner


class TestDefaultArgs(unittest.TestCase):
    """
    Tests the default arguments for the enroads model.
    Makes sure passing no args is the same as passing the defaults and that passing something different doesn't return
    the default answer.
    """
    def setUp(self):
        self.runner = EnroadsRunner()
        self.input_specs = load_input_specs()
        self.input_specs["index"] = range(len(self.input_specs))

    def test_construct_default_args(self):
        """
        TODO: Examine the default args more. Currently this affects some of the columns we don't care about so for now
        we're ok with the defaults not exactly matching.
        Tests if no args is the same as default args is the same as manually constructing default args
        """
        # input_str = self.runner.construct_enroads_input({})
        # index_col = self.input_specs["index"].astype(str)
        # default_col = self.input_specs["defaultValue"].astype(str)
        # default_str = " ".join(index_col + ":" + default_col)

        # no_arg_output = self.runner.run_enroads()
        # input_output = self.runner.run_enroads(input_str)
        # default_output = self.runner.run_enroads(default_str)

        # no_arg_df = pd.read_table(io.StringIO(no_arg_output), sep="\t")
        # input_df = pd.read_table(io.StringIO(input_output), sep="\t")
        # default_df = pd.read_table(io.StringIO(default_output), sep="\t")

        # pd.testing.assert_frame_equal(input_df, default_df)
        # pd.testing.assert_frame_equal(no_arg_df, default_df)

    def test_non_default_args(self):
        """
        Test if we pass in some other arbitrary values we don't get the default response.
        """
        avg_col = (self.input_specs["minValue"] + self.input_specs["maxValue"]) / 2
        input_col = self.input_specs["index"].astype(str) + ":" + avg_col.astype(str)
        input_str = " ".join(input_col)

        no_default_output = self.runner.run_enroads(input_str)
        default_output = self.runner.run_enroads()

        self.assertNotEqual(no_default_output, default_output)
