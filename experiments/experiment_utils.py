"""
Utility functions for experimentation
"""
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from presp.prescriptor import PrescriptorFactory, NNPrescriptorFactory
import yaml

from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.candidates.direct import DirectFactory
from evolution.evaluation.evaluator import EnROADSEvaluator
from evolution.utils import process_config


class Experimenter:
    """
    Abstract experimenter class that handles the loading of candidates and their evaluation. Does pretty much all of
    the heavy lifting and just requires the creation of a PrescriptorFactory to be implemented.
    NOTE: Caches the results of candidate evaluation which could be insanely memory intensive if there are a lot
    of contexts.
    """
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        with open(results_dir / "config.yml", "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.config = process_config(self.config)

        self.prescriptor_factory = self.create_prescriptor_factory(self.config)
        self.population = self.prescriptor_factory.load_population(self.results_dir / "population")

        self.evaluator = EnROADSEvaluator(self.config["context"],
                                          self.config["actions"],
                                          self.config["outcomes"],
                                          n_jobs=1,
                                          batch_size=self.config["batch_size"],
                                          device=self.config["device"],
                                          decomplexify=self.config.get("decomplexify", False))

        self.results_cache = {}

    @abstractmethod
    def create_prescriptor_factory(self, config: dict) -> PrescriptorFactory:
        """
        Creates a prescriptor factory based on the config.
        """
        raise NotImplementedError("Must implement prescriptor factory creation in subclass")

    def get_candidate_results(self, cand_id: str) -> tuple[list[dict], list[pd.DataFrame], np.ndarray]:
        """
        Gets the context_actions dicts, outcomes_dfs, and metrics for a candidate.
        """
        # If the results are already cached, just grab them
        if cand_id in self.results_cache:
            return self.results_cache[cand_id]

        # The full process of evaluation on a candidate
        candidate = self.population[cand_id]
        context_actions_dicts = self.evaluator.prescribe_actions(candidate)
        outcomes_dfs = self.evaluator.run_enroads(context_actions_dicts)
        metrics = self.evaluator.compute_metrics(context_actions_dicts, outcomes_dfs)

        # Cache results for future use
        self.results_cache[cand_id] = (context_actions_dicts, outcomes_dfs, metrics)

        # Return copies to avoid in-place modifications
        context_actions_dicts = [dict(context_actions_dict) for context_actions_dict in context_actions_dicts]
        outcomes_dfs = [outcomes_df.copy() for outcomes_df in outcomes_dfs]
        metrics = metrics.copy()
        return context_actions_dicts, outcomes_dfs, metrics


class NNExperimenter(Experimenter):
    """
    Implements the experimenter for the NNPrescriptor.
    """
    def create_prescriptor_factory(self, config: dict) -> NNPrescriptorFactory:
        return NNPrescriptorFactory(EnROADSPrescriptor,
                                    model_params=config["model_params"],
                                    device=config["device"],
                                    actions=config["actions"])


class DirectExperimenter(Experimenter):
    """
    Implements the experimenter for the DirectPrescriptor.
    """
    def create_prescriptor_factory(self, config: dict) -> DirectFactory:
        return DirectFactory(config["actions"])
