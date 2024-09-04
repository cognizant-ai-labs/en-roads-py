"""
PyTorch implementation of NSGA-II.
"""
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from evolution.candidate import Candidate
from evolution.crossover.uniform_crossover import UniformCrossover
from evolution.mutation.uniform_mutation import UniformMutation
from evolution.evaluation.evaluator import Evaluator
from evolution.sorting.distance_calculation.crowding_distance import CrowdingDistanceCalculator
from evolution.sorting.nsga2_sorter import NSGA2Sorter
from evolution.parent_selection.tournament_selector import TournamentSelector


class Evolution():
    """
    Class handling the overall NSGA-II evolutionary loop.
    Takes in a config file that determines parent selection, mutation, crossover, distance calcuation, sorting,
    and evaluation.
    Saves the config file and intermediate candidates + results to disk.
    """
    def __init__(self, config: dict):
        self.save_path = Path(config["save_path"])
        self.save_path.mkdir(parents=True, exist_ok=False)
        with open(self.save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        # Evolution params
        self.evolution_params = config["evolution_params"]
        self.pop_size = self.evolution_params["pop_size"]
        self.n_generations = self.evolution_params["n_generations"]
        self.n_elites = self.evolution_params["n_elites"]
        self.seed_path = None
        if "seed_params" in config and "seed_path" in config["seed_params"]:
            self.seed_path = Path(config["seed_params"]["seed_path"])

        self.parent_selector = TournamentSelector(config["remove_population_pct"])
        self.mutator = UniformMutation(config["mutation_factor"], config["mutation_rate"])
        self.crossover = UniformCrossover(mutator=self.mutator)
        distance_calculator = CrowdingDistanceCalculator()
        self.sorter = NSGA2Sorter(distance_calculator)

        self.model_params = config["model_params"]
        self.actions = config["actions"]
        self.outcomes = config["outcomes"]

        self.evaluator = Evaluator(**config["eval_params"])

    def make_new_pop(self, candidates: list[Candidate], n: int, gen: int) -> list[Candidate]:
        """
        Creates new population of candidates.
        Doesn't remove any candidates from the previous generation if we're on the first generation.
        """
        first_selector = TournamentSelector(0)
        selector = self.parent_selector if gen > 1 else first_selector
        children = []
        while len(children) < n:
            parents = selector.select_parents(candidates)
            children.extend(self.crossover.crossover(f"{gen}_{len(children)}",
                                                     parents[0],
                                                     parents[1]))
        return children

    def seed_first_gen(self):
        """
        Creates the first generation by taking seeds and creating an average of them.
        """
        candidates = []
        if self.seed_path:
            print("Seeding from ", self.seed_path)
            for seed in self.seed_path.iterdir():
                candidate = Candidate.from_seed(seed, self.model_params, self.actions, self.outcomes)
                candidates.append(candidate)

        print("Generating random seed generation")
        i = len(candidates)
        while i < self.pop_size:
            candidate = Candidate(f"0_{i}", [], self.model_params, self.actions, self.outcomes)
            candidates.append(candidate)
            i += 1

        self.evaluator.evaluate_candidates(candidates)
        candidates = self.sorter.sort_candidates(candidates)

        self.record_gen_results(0, candidates)
        return candidates

    def record_gen_results(self, gen: int, candidates: list[Candidate]):
        """
        Logs results of generation to CSV. Saves candidates to disk
        """
        gen_results = [c.record_state() for c in candidates]
        gen_results_df = pd.DataFrame(gen_results)
        csv_path = self.save_path / f"{gen}.csv"
        gen_results_df.to_csv(csv_path, index=False)
        for c in candidates:
            if c.cand_id.startswith(str(gen)):
                c.save(self.save_path / str(gen) / f"{c.cand_id}.pt")

    def neuroevolution(self):
        """
        Main Neuroevolution Loop that performs NSGA-II.
        After initializing the first population randomly, goes through 3 steps in each generation:
        1. Evaluate candidates
        2. Select parents
        2a Log performance of parents
        3. Make new population from parents
        """
        print("Beginning evolution...")
        sorted_parents = self.seed_first_gen()
        offspring = []
        for gen in tqdm(range(1, self.n_generations+1)):
            # Create offspring
            keep = min(self.n_elites, len(sorted_parents))
            offspring = self.make_new_pop(sorted_parents, self.pop_size-keep, gen)

            # Add elites to new generation
            # NOTE: The elites are also passed into the evaluation function. Make sure your
            # evaluator can handle this!
            new_parents = sorted_parents[:keep] + offspring
            self.evaluator.evaluate_candidates(new_parents)

            # Set rank and distance of parents
            sorted_parents = self.sorter.sort_candidates(new_parents)

            # Record the performance of the most successful candidates
            self.record_gen_results(gen, sorted_parents)

        return sorted_parents
