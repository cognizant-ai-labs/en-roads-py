import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from evolution.candidate import EnROADSPrescriptor
from evolution.evaluator import EnROADSEvaluator


class NoveltyEvaluator(EnROADSEvaluator):
    def __init__(self, context: list[str], actions: list[str]):
        super().__init__(context=context, actions=actions, outcomes={"novelty": False})

        self.idx_map = {}
        self.index = faiss.IndexFlatL2(111)
        self.k = 10

    def _novelty_algorithm(self, population: np.ndarray, indices: np.ndarray, index: faiss.IndexFlatL2, k: int):
        D, I = index.search(population, k+1)

        already_there = I[:, 0] == indices

        novelty_old = np.mean(D[already_there, 1:k+1], axis=1)
        novelty_new = np.mean(D[~already_there, :k], axis=1)

        total_novelty = np.empty(population.shape[0])
        total_novelty[already_there] = novelty_old
        total_novelty[~already_there] = novelty_new

        to_add = population[~already_there]
        index.add(to_add)

        return total_novelty

    def _get_average_df(self, dfs: list[pd.DataFrame]):
        avg_df = dfs[0].copy()
        for df in dfs[1:]:
            avg_df += df
        avg_df /= len(dfs)
        return avg_df

    def evaluate_subset(self, population: list[EnROADSPrescriptor], verbose=1) -> list[np.ndarray]:
        """
        TODO: This can be parallelized.
        """
        population_behavior = []
        indices = []
        i = len(self.idx_map)
        for candidate in tqdm(population, leave=False, desc="Evaluating population"):
            context_actions_dicts = self.prescribe_actions(candidate)
            outcomes_dfs = self.run_enroads(context_actions_dicts)

            avg_outcomes = self._get_average_df(outcomes_dfs)["Temperature change from 1850"]
            population_behavior.append(avg_outcomes.to_numpy())

            if candidate.cand_id not in self.idx_map:
                self.idx_map[candidate.cand_id] = i
                i += 1
            indices.append(self.idx_map[candidate.cand_id])

        population_behavior = np.stack(population_behavior).astype(np.float32)

        novelties = self._novelty_algorithm(population_behavior, indices, self.index, self.k)
        novelties = np.array(novelties).reshape(-1, 1)
        novelties = [-1 * novelty for novelty in novelties]
        return novelties
    
    def evaluate_population(self, population: list[EnROADSPrescriptor], force=True, verbose=1):
        """
        Evaluates an entire population of prescriptors.
        Doesn't evaluate prescriptors that already have metrics unless force is True.
        Sets candidates' metrics and outcomes for evolution.
        :param population: The population of prescriptors to evaluate.
        :param force: Whether to force evaluation of all prescriptors.
        :param verbose: Whether to show a progress bar for sequential evaluation.
        """
        pop_subset = [cand for cand in population]
        pop_results = self.evaluate_subset(pop_subset, verbose=verbose)
        for candidate, metrics in zip(pop_subset, pop_results):
            candidate.metrics = metrics
            candidate.outcomes = self.outcomes
