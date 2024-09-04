"""
Custom problem for PyMoo to optimize En-ROADS.
"""
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from enroadspy import load_input_specs
from enroadspy.enroads_runner import EnroadsRunner
from evolution.outcomes.outcome_manager import OutcomeManager


class NoveltyProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for En-ROADS.
    All sliders have an upper and lower bound defined by inputSpecs. All switches range between 0 and 1.
        Switches are snapped to their correct values at evaluation.
    Constraints are set on start and stop years such that start year is less than stop year.
    All outcomes are minimized so we have to pre and post process them.
    """
    def __init__(self, actions: list[str], outcomes: dict[str, bool], k: int = 3):
        self.input_specs = load_input_specs()
        xl = np.zeros(len(actions))
        xu = np.ones(len(actions))
        switch_idxs = []
        switchl = []
        switchu = []
        self.start_year_idxs = set()
        for i, action in enumerate(actions):
            row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
            if row["kind"] == "slider":
                xl[i] = row["minValue"]
                xu[i] = row["maxValue"]
                if "start_time" in action and "stop_time" in actions[i+1]:
                    self.start_year_idxs.add(i)
            else:
                switch_idxs.append(i)
                switchl.append(row["offValue"])
                switchu.append(row["onValue"])

        super().__init__(n_var=len(actions), n_obj=1, n_ieq_constr=len(self.start_year_idxs), xl=xl, xu=xu)

        # To evaluate candidate solutions
        self.runner = EnroadsRunner()
        self.actions = list(actions)
        self.outcomes = dict(outcomes.items())
        self.outcome_manager = OutcomeManager(list(self.outcomes.keys()))

        # To parse switches
        self.switch_idxs = switch_idxs
        self.switchl = switchl
        self.switchu = switchu

        # Novelty search parameters
        self.k = k
        self.archive, self.threshold, self.knn, self.scaler = self.initialize_archive(100)

    def initialize_archive(self, n: int) -> tuple[np.ndarray, float, NearestNeighbors, StandardScaler]:
        """
        Initializes our parameters for the novelty search.
        Creates a random population size ~n and evaluates them to get behaviors.
        Trains a scaler on the behaviors and normalizes them.
        Finds the median knn distance and sets the threshold novelty score to it.
        TODO: Make the threshold dynamic over time.
        """
        # Generate random population of candidates
        rand_pop = []
        while len(rand_pop) < n:
            rand_cand = np.random.uniform(self.xl, self.xu, len(self.xl))
            for idx in self.start_year_idxs:
                if rand_cand[idx] >= rand_cand[idx+1]:
                    continue
            rand_pop.append(rand_cand)
        rand_pop = np.array(rand_pop)

        # Evaluate random population
        behaviors = []
        for x in rand_pop:
            actions_dict = self.params_to_actions_dict(x)
            outcomes_df = self.runner.evaluate_actions(actions_dict)
            results_dict = self.outcome_manager.process_outcomes(actions_dict, outcomes_df)
            results = np.array([results_dict[outcome] for outcome in self.outcomes])
            behaviors.append(results)
        behaviors = np.array(behaviors)

        # Normalize behaviors
        scaler = StandardScaler()
        archive = scaler.fit_transform(behaviors)

        # Find the median knn distance and set our threshold to it
        knn = NearestNeighbors(n_neighbors=self.k).fit(archive)
        distances, _ = knn.kneighbors(archive)
        threshold = max(distances[:, -1])
        return archive, threshold, knn, scaler

    def compute_novelty(self, results_dict: dict[str, float]) -> float:
        """
        Compute novelty of a candidate solution's behavior with respect to the archive.
        If there are less than k points in the archive, add the candidate to the archive.
        If the candidate exceeds a threshold distance from its k nearest neighbors, add it to the archive.
        """
        results = np.array([[results_dict[outcome] for outcome in self.outcomes]])
        if np.isnan(results).any():
            return -1
        transformed = self.scaler.transform(results)
        distances, _ = self.knn.kneighbors(transformed)
        novelty = np.mean(distances[:, -1])

        if novelty > self.threshold:
            self.archive = np.vstack([self.archive, results])
            self.knn.fit(self.archive)

        return novelty

    def params_to_actions_dict(self, x) -> dict[str, float]:
        """
        Converts the optimized parameters into an actions dict readable by our EnroadsRunner.
        """
        parsed = x.copy()
        parsed[self.switch_idxs] = np.where(parsed[self.switch_idxs] < 0.5, self.switchl, self.switchu)
        actions_dict = dict(zip(self.actions, parsed))
        return actions_dict

    def _evaluate(self, x, out, *args, **kwargs):
        actions_dict = self.params_to_actions_dict(x)
        outcomes_df = self.runner.evaluate_actions(actions_dict)
        results_dict = self.outcome_manager.process_outcomes(actions_dict, outcomes_df)

        f = -1 * self.compute_novelty(results_dict)

        g = [x[idx] - x[idx+1] for idx in self.start_year_idxs]

        out["F"] = [f]
        out["G"] = g
