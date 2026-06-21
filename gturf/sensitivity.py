"""
Sensitivity analysis for the GA hyperparameters.

Sweeps one hyperparameter at a time (crossover rate, mutation rate, maximum
generations, population cap) around the configured defaults and records the
achieved reach, so a reviewer can see how robust the results are to parameter
choices. Designed to run on the candidate sets already produced by the main
pipeline, reusing the incidence matrices for speed.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import GTurfConfig
from .pipeline import PipelineResults
from .turf import build_incidence_matrix
from .ga import run_ga


DEFAULT_GRIDS = {
    "crossover_rate": [0.5, 0.6, 0.7, 0.8, 0.9],
    "mutation_rate": [0.05, 0.15, 0.25, 0.35],
    "generations": [10, 20, 30, 40, 50, 60],
    "max_pop": [500, 1000, 2000, 4000, 8000],
}


def run_sensitivity(
    results: PipelineResults,
    cfg: GTurfConfig,
    r: int,
    parameters: Optional[List[str]] = None,
    grids: Optional[Dict[str, List]] = None,
    runs_per_setting: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """One-at-a-time sensitivity sweep at a fixed bundle size ``r``.

    For each occupation, each parameter in ``parameters``, and each value in its
    grid, the GA is run ``runs_per_setting`` times and the best/mean reach is
    recorded. All other parameters stay at the ``cfg`` defaults.
    """
    parameters = parameters or list(DEFAULT_GRIDS.keys())
    grids = {**DEFAULT_GRIDS, **(grids or {})}
    rows = []

    for occ in results.candidate_skills:
        candidate_ids = results.candidate_skills[occ]
        turf_df = results.turf_df[occ]
        if len(candidate_ids) <= r or len(turf_df) == 0:
            continue
        X = build_incidence_matrix(turf_df, candidate_ids)
        n_jobs = len(turf_df)

        for param in parameters:
            for value in grids[param]:
                kwargs = dict(
                    generations=cfg.generations,
                    crossover_rate=cfg.crossover_rate,
                    mutation_rate=cfg.mutation_rate,
                    elitism=cfg.elitism,
                    early_stop_patience=cfg.early_stop_patience,
                    min_delta=cfg.min_delta,
                    init_frac=cfg.init_frac,
                    max_pop=cfg.max_pop,
                    n_jobs=n_jobs,
                )
                kwargs[param] = value
                reaches, gens = [], []
                for run_i in range(runs_per_setting):
                    bdf = run_ga(
                        X, candidate_ids, results.label_map, r,
                        seed=cfg.base_seed + run_i, **kwargs,
                    )
                    reaches.append(float(bdf["best_so_far_reach"].iloc[-1]))
                    gens.append(int(bdf["generation"].iloc[-1]))
                arr = np.array(reaches)
                rows.append({
                    "occupation": occ,
                    "r": r,
                    "parameter": param,
                    "value": value,
                    "best_reach": float(arr.max()),
                    "mean_reach": float(arr.mean()),
                    "std_reach": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "mean_pct": round(arr.mean() / n_jobs * 100, 3),
                    "mean_generations": float(np.mean(gens)),
                })
                if verbose:
                    print(f"  {occ} | {param}={value} | "
                          f"mean reach {arr.mean()/n_jobs*100:.2f}% "
                          f"(gens {np.mean(gens):.1f})")

    return pd.DataFrame(rows)


def summarise_sensitivity(sens_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(occupation, parameter) range of mean reach %, i.e. how much the
    parameter moved the outcome. Small ranges => robust."""
    if sens_df.empty:
        return sens_df
    g = sens_df.groupby(["occupation", "parameter"])["mean_pct"]
    out = g.agg(["min", "max"]).reset_index()
    out["range_pp"] = (out["max"] - out["min"]).round(3)
    return out.sort_values(["occupation", "range_pp"], ascending=[True, False])
