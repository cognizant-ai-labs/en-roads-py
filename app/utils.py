import json

import dill
import pandas as pd

from moo.problems.nn_problem import NNProblem
from enroads_runner import EnroadsRunner
from evolution.outcomes.outcome_manager import OutcomeManager


def load_outcomes_and_metrics_dfs():
    save_path = "results/pymoo/context"
    with open(save_path + "/config.json", 'r') as f:
        config = json.load(f)

    actions = config["actions"]
    outcomes = config["outcomes"]
    with open(save_path + "/results", 'rb') as f:
        res = dill.load(f)
        print("Loaded Checkpoint:", res)

    X = res.X
    F = res.F

    context_df = pd.read_csv("experiments/scenarios/gdp_context.csv")
    context_df = context_df.drop(columns=["F", "scenario"])
    problem = NNProblem(context_df,
                        {"in_size": len(context_df.columns), "hidden_size": 16, "out_size": len(actions)},
                        actions,
                        outcomes)

    runner = EnroadsRunner("app/temp")
    baseline_df = runner.evaluate_actions({})
    outcome_manager = OutcomeManager(list(outcomes.keys()))
    baseline_metrics = outcome_manager.process_outcomes({}, baseline_df)

    all_outcomes_dfs = []
    all_metrics = []
    for cand_idx in range(X.shape[0]):
        context_actions_dicts = problem.params_to_context_actions_dicts(X[cand_idx])
        cand_outcomes_dfs = problem.run_enroads(context_actions_dicts)
        all_outcomes_dfs.append(cand_outcomes_dfs)
        for i, (context_actions_dict, outcome_df) in enumerate(zip(context_actions_dicts, cand_outcomes_dfs)):
            metrics = outcome_manager.process_outcomes(context_actions_dict, outcome_df)
            metrics["cand_idx"] = cand_idx
            metrics["context_idx"] = i
            all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics), all_outcomes_dfs, baseline_metrics, baseline_df