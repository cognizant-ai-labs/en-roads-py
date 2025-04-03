"""
Utility functions for experimentation
"""
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from presp.prescriptor import Prescriptor, PrescriptorFactory, NNPrescriptorFactory
import yaml

from evolution.candidates.candidate import EnROADSPrescriptor
from evolution.candidates.direct import DirectFactory
from evolution.evaluation.evaluator import EnROADSEvaluator


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

        self.prescriptor_factory = self.create_prescriptor_factory(self.config)

        self.evaluator = EnROADSEvaluator(self.config["context"],
                                          self.config["actions"],
                                          self.config["outcomes"],
                                          n_jobs=1,
                                          batch_size=self.config["batch_size"],
                                          device=self.config["device"])
        
        self.results_cache = {}

    @abstractmethod
    def create_prescriptor_factory(self, config: dict) -> PrescriptorFactory:
        """
        Creates a prescriptor factory based on the config.
        """
        raise NotImplementedError("Must implement prescriptor factory creation in subclass")

    def get_candidate_from_id(self, cand_id: str) -> Prescriptor:
        """
        Loads a candidate from an id.
        NOTE: The seeds get saved in generation 1 but are indicated by starting with 0 so we hard-code their loading.
        """
        gen = cand_id.split("_")[0]
        if gen == 0:
            gen += 1
        cand_path = self.results_dir / cand_id.split("_")[0] / f"{cand_id}"
        return self.prescriptor_factory.load(cand_path)

    def get_candidate_results(self, cand_id: str) -> tuple[list[dict], list[pd.DataFrame], np.ndarray]:
        """
        Gets the context_actions dicts, outcomes_dfs, and metrics for a candidate.
        """

        # If the results are already cached, just grab them
        if cand_id in self.results_cache:
            return self.results_cache[cand_id]

        # The full process of evaluation on a candidate
        candidate = self.get_candidate_from_id(cand_id)
        context_actions_dicts = self.evaluator.prescribe_actions(candidate)
        outcomes_dfs = self.evaluator.run_enroads(context_actions_dicts)
        metrics = self.evaluator.compute_metrics(context_actions_dicts, outcomes_dfs)

        # Cache results for future use
        self.results_cache[cand_id] = (context_actions_dicts, outcomes_dfs, metrics)

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
