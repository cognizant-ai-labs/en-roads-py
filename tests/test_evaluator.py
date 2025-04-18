"""
Unit tests for the evaluator class.
"""
import unittest

import numpy as np
import pandas as pd
from presp.prescriptor import NNPrescriptorFactory
import torch

from enroadspy import load_input_specs
from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.evaluation.evaluator import EnROADSEvaluator


class TestEvaluator(unittest.TestCase):
    """
    Class testing the evaluator class. This is the most important part of the experiment as it executes the
    en-roads model. We need to make sure all our inputs and outputs are correct.
    """
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        # Dummy config to test with
        config = {
            "evolution_params": {
                "n_generations": 100,
                "population_size": 100,
                "remove_population_pct": 0.8,
                "n_elites": 10,
                "mutation_rate": 0.1,
                "mutation_factor": 0.1,
                "save_path": "tests/temp"
            },
            "device": "cpu",
            "batch_size": 64,
            "n_jobs": 1,
            "model_params": [
                {"type": "linear", "in_features": 4, "out_features": 64},
                {"type": "tanh"},
                {"type": "linear", "in_features": 64, "out_features": 115},
                {"type": "sigmoid"}
            ],
            "context": [
                "_global_population_in_2100",
                "_long_term_gdp_per_capita_rate",
                "_near_term_gdp_per_capita_rate",
                "_transition_time_to_reach_long_term_gdp_per_capita_rate"
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
                "_performance_standard_time",
                "_electric_carrier_subsidy_with_required_comp_assets",
                "_switch_to_use_transport_electrification_detailed_settings",
                "_electric_carrier_subsidy_transport",
                "_percent_of_required_elec_complementary_assets_to_build",
                "_electric_carrier_subsidy_end_year_transport",
                "_cap_fuel_powered_road_and_rail_transport",
                "_time_to_achieve_electrification_target_transport_road_and_rail",
                "_year_starting_electrification_policy_transport",
                "_cap_fuel_powered_shipping_and_aviation_transport",
                "_time_to_achieve_electrification_target_air_and_water_transport",
                "_year_starting_electrification_policy_air_and_water_transport",
                "_electric_carrier_subsidy_stationary",
                "_electric_carrier_subsidy_end_year_stationary",
                "_cap_fuel_powered_stationary",
                "_time_to_achieve_electrification_target_stationary",
                "_year_starting_electrification_policy_stationary",
                "_target_change_in_other_ghgs_for_ag",
                "_use_detailed_food_and_ag_controls",
                "_target_change_in_other_ghgs_for_ls",
                "_target_change_in_other_ghgs_for_crops",
                "_start_year_for_ag_practice_adoption",
                "_time_to_achieve_ag_practice_targets",
                "_land_cdr_percent_of_reference",
                "_choose_nature_cdr_by_type",
                "_percent_available_land_for_afforestation",
                "_afforestation_cdr_start_year",
                "_years_to_secure_land_for_afforestation",
                "_years_to_plant_land_committed_to_afforestation",
                "_ag_soil_carbon_percent_of_max_cdr_achieved",
                "_agricultural_soil_carbon_start_year",
                "_biochar_percent_of_max_cdr_achieved",
                "_biochar_start_year",
                "_target_change_other_ghgs_leakage_and_waste",
                "_use_detailed_other_ghg_controls",
                "_target_change_other_ghgs_energy",
                "_target_change_other_ghgs_waste",
                "_target_change_other_gas_industry",
                "_target_change_f_gas",
                "_other_ghg_emissions_change_start_year",
                "_time_to_achieve_other_ghg_changes",
                "_deforestation_slider_setting",
                "_switch_use_land_detailed_settings",
                "_target_reduction_in_deforestation",
                "_start_year_of_deforestation_reduction",
                "_years_to_achieve_deforestation_policy",
                "_target_reduction_in_mature_forest_degradation",
                "_start_year_of_mature_forest_degradation_reduction",
                "_years_to_achieve_mature_forest_degradation_policy",
                "_tech_cdr_percent_of_reference",
                "_choose_cdr_by_type",
                "_dac_percent_of_max_cdr_achieved",
                "_direct_air_capture_start_year",
                "_mineralization_percent_of_max_cdr_achieved",
                "_mineralization_start_year"
            ],
            "outcomes": {
                "Temperature above 1.5C": True,
                "Max cost of energy": True
            },
        }

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

        actions_dict = {"_source_subsidy_delivered_coal_tce": 100}
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

        actions_dict = {"_source_subsidy_delivered_coal_tce": 100}
        enroads_input = self.evaluator.enroads_runner.construct_enroads_input(actions_dict)
        split_input = enroads_input.split(" ")
        for i, (default, inp) in enumerate(zip(vals, split_input)):
            _, inp_val = inp.split(":")
            if i == 0:
                self.assertTrue(np.isclose(float(inp_val), 100), "Didn't modify first input correctly")
            else:
                self.assertTrue(np.isclose(float(inp_val), default), "Messed up first input")

        actions_dict = {"_source_subsidy_start_time_delivered_coal": 2040}
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
            row = input_specs[input_specs["varId"] == action]
            if row["kind"].iloc[0] == "switch":
                true_actions[action] = 1
        true_outcomes = self.evaluator.enroads_runner.evaluate_actions(true_actions)

        # Switch all checkboxes to false
        false_actions = {}
        for action in self.config["actions"]:
            row = input_specs[input_specs["varId"] == action]
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
            row = input_specs[input_specs["varId"] == action].iloc[0]
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
        exceptions = ['_apply_carbon_tax_to_biofuels',
                      '_ccs_carbon_tax_qualifier',
                      '_qualifying_path_nuclear',
                      '_qualifying_path_bioenergy',
                      '_qualifying_path_fossil_ccs',
                      '_qualifying_path_gas']
        self.assertEqual(set(bad_actions), set(exceptions), "Switches besides exceptions changed the past")

    def test_sliders_change_past(self):
        """
        Checks to see if setting the slider to the min or max value changes the past.
        """
        input_specs = load_input_specs()
        baseline = self.evaluator.enroads_runner.evaluate_actions({})
        bad_actions = []
        # TODO: When we set this to input_specs['varId'].unique() we get some fails we need to account for.
        for action in self.config["actions"]:
            row = input_specs[input_specs["varId"] == action].iloc[0]
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
        actions = ["_source_subsidy_delivered_coal_tce",
                   "_source_subsidy_start_time_delivered_coal",
                   "_source_subsidy_stop_time_delivered_coal",]

        input_specs = load_input_specs()
        min_time = input_specs[input_specs["varId"] == actions[1]].iloc[0]["minValue"]
        max_time = input_specs[input_specs["varId"] == actions[1]].iloc[0]["maxValue"]

        both_start_dict = {actions[0]: -15, actions[1]: min_time, actions[2]: min_time}
        both_end_dict = {actions[0]: -15, actions[1]: max_time, actions[2]: max_time}
        crossed_dict = {actions[0]: -15, actions[1]: max_time, actions[2]: min_time}

        both_start_outcomes = self.evaluator.enroads_runner.evaluate_actions(both_start_dict)
        both_end_outcomes = self.evaluator.enroads_runner.evaluate_actions(both_end_dict)
        crossed_outcomes = self.evaluator.enroads_runner.evaluate_actions(crossed_dict)

        self.assertTrue(both_start_outcomes.equals(both_end_outcomes))
        self.assertTrue(both_start_outcomes.equals(crossed_outcomes))
    

class TestDecomplexify(unittest.TestCase):
    def setUp(self):
        config = {
            "evolution_params": {
            },
            "device": "cpu",
            "batch_size": 64,
            "n_jobs": -1,
            "decomplexify": True,
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
                "_performance_standard_time",
                "_electric_carrier_subsidy_transport",
                "_percent_of_required_elec_complementary_assets_to_build",
                "_electric_carrier_subsidy_end_year_transport",
                "_cap_fuel_powered_road_and_rail_transport",
                "_time_to_achieve_electrification_target_transport_road_and_rail",
                "_year_starting_electrification_policy_transport",
                "_cap_fuel_powered_shipping_and_aviation_transport",
                "_time_to_achieve_electrification_target_air_and_water_transport",
                "_year_starting_electrification_policy_air_and_water_transport",
                "_electric_carrier_subsidy_stationary",
                "_electric_carrier_subsidy_end_year_stationary",
                "_cap_fuel_powered_stationary",
                "_time_to_achieve_electrification_target_stationary",
                "_year_starting_electrification_policy_stationary",
                "_target_change_in_other_ghgs_for_ls",
                "_target_change_in_other_ghgs_for_crops",
                "_start_year_for_ag_practice_adoption",
                "_time_to_achieve_ag_practice_targets",
                "_choose_nature_cdr_by_type",
                "_percent_available_land_for_afforestation",
                "_afforestation_cdr_start_year",
                "_years_to_secure_land_for_afforestation",
                "_years_to_plant_land_committed_to_afforestation",
                "_ag_soil_carbon_percent_of_max_cdr_achieved",
                "_agricultural_soil_carbon_start_year",
                "_biochar_percent_of_max_cdr_achieved",
                "_biochar_start_year",
                "_target_change_other_ghgs_energy",
                "_target_change_other_ghgs_waste",
                "_target_change_other_gas_industry",
                "_target_change_f_gas",
                "_other_ghg_emissions_change_start_year",
                "_time_to_achieve_other_ghg_changes",
                "_target_reduction_in_deforestation",
                "_start_year_of_deforestation_reduction",
                "_years_to_achieve_deforestation_policy",
                "_target_reduction_in_mature_forest_degradation",
                "_start_year_of_mature_forest_degradation_reduction",
                "_years_to_achieve_mature_forest_degradation_policy",
                "_dac_percent_of_max_cdr_achieved",
                "_direct_air_capture_start_year",
                "_mineralization_percent_of_max_cdr_achieved",
                "_mineralization_start_year"
            ],
            "outcomes": {
                "Temperature change from 1850": True,
                "Actions taken": True,
                "Action magnitude": True
            }
        }
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
