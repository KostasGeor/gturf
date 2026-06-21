"""
Genetic algorithm for TURF subset selection.

Each chromosome is a length-M binary vector with exactly r active positions
(a skill bundle). Fitness is the unduplicated reach (see :mod:`gturf.turf`).
Operators: fitness-proportionate (roulette) selection, uniform crossover, swap
mutation (which preserves the cardinality constraint), elitism, and a repair
operator that restores |bundle| = r after crossover.

The search stops at ``generations`` or when reach has not improved by at least
``min_delta`` for ``early_stop_patience`` consecutive generations.
"""
from __future__ import annotations

import random
from math import comb
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .turf import vectorized_reach, bundle_label


def sample_initial_population(
    n_skills: int,
    r: int,
    init_frac: float,
    max_pop: int,
    seed: int,
) -> List[np.ndarray]:
    """Sample distinct random r-hot binary vectors as the initial population,
    sized at ``init_frac`` of the full space but capped at ``max_pop``.
    """
    rng = random.Random(seed)
    total = comb(n_skills, r)
    k = min(max_pop, max(4, int(np.floor(total * init_frac))))
    seen = set()
    pop: List[np.ndarray] = []
    # Cap attempts to avoid pathological loops when k approaches `total`.
    attempts = 0
    max_attempts = k * 50 + 1000
    while len(pop) < k and attempts < max_attempts:
        idx = tuple(sorted(rng.sample(range(n_skills), r)))
        attempts += 1
        if idx in seen:
            continue
        seen.add(idx)
        vec = np.zeros(n_skills, dtype=np.int8)
        for i in idx:
            vec[i] = 1
        pop.append(vec)
    return pop


def run_ga(
    X: np.ndarray,
    skill_ids: List[str],
    label_map: Dict[str, str],
    r: int,
    *,
    generations: int = 40,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.25,
    elitism: int = 2,
    seed: int = 42,
    early_stop_patience: int = 8,
    min_delta: int = 1,
    init_frac: float = 1 / 3,
    max_pop: int = 8000,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Run the GA for a fixed bundle size r. Returns a per-generation history
    dataframe whose last row holds the best bundle found.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    n = len(skill_ids)
    if n_jobs is None:
        n_jobs = X.shape[0]

    population = sample_initial_population(n, r, init_frac, max_pop, seed)
    pop_size = len(population)
    if pop_size < 4:
        raise ValueError(f"Initial population too small ({pop_size}) for r={r}.")

    def repair(vec: np.ndarray) -> np.ndarray:
        out = vec.copy()
        ones = np.where(out == 1)[0].tolist()
        zeros = np.where(out == 0)[0].tolist()
        while len(ones) > r:
            i = rng.choice(ones); out[i] = 0; ones.remove(i); zeros.append(i)
        while len(ones) < r:
            i = rng.choice(zeros); out[i] = 1; zeros.remove(i); ones.append(i)
        return out

    def crossover(p1: np.ndarray, p2: np.ndarray):
        mask = np.random.random(n) < 0.5
        c1, c2 = p1.copy(), p2.copy()
        c1[mask] = p2[mask]; c2[mask] = p1[mask]
        return c1, c2

    def mutate(vec: np.ndarray) -> np.ndarray:
        out = vec.copy()
        ones = np.where(out == 1)[0]
        zeros = np.where(out == 0)[0]
        if len(ones) and len(zeros):
            out[int(rng.choice(ones))] = 0
            out[int(rng.choice(zeros))] = 1
        return out

    best_rows = []
    best_so_far = -1
    no_improve = 0

    for gen in range(generations):
        fitness = vectorized_reach(X, population)
        best_idx = int(np.argmax(fitness))
        best_reach = int(fitness[best_idx])
        best_vec = population[best_idx]

        if best_reach >= best_so_far + min_delta:
            best_so_far = best_reach
            no_improve = 0
        else:
            no_improve += 1

        row = {
            "generation": gen + 1,
            "Reach": best_reach,
            "Reach %": round(best_reach / n_jobs * 100, 2) if n_jobs else 0.0,
            "Combination Number": r,
            "Skills": bundle_label(best_vec, skill_ids, label_map),
            "best_so_far_reach": best_so_far,
            "no_improve_streak": no_improve,
        }
        for i, sid in enumerate(skill_ids):
            row[sid] = int(best_vec[i])
        best_rows.append(row)

        if no_improve >= early_stop_patience:
            break

        # selection probabilities
        f = fitness.astype(np.float64)
        s = f.sum()
        probs = f / s if s > 0 else np.ones(pop_size) / pop_size
        probs = probs / probs.sum()

        elite_idx = np.argsort(fitness)[::-1][:elitism]
        new_pop = [population[i].copy() for i in elite_idx]

        n_pairs = pop_size + 4
        p1_idx = np.random.choice(pop_size, size=n_pairs, p=probs)
        p2_idx = np.random.choice(pop_size, size=n_pairs, p=probs)
        cx = np.random.random(n_pairs) < crossover_rate
        m1 = np.random.random(n_pairs) < mutation_rate
        m2 = np.random.random(n_pairs) < mutation_rate

        pi = 0
        while len(new_pop) < pop_size:
            a = population[p1_idx[pi]].copy()
            b = population[p2_idx[pi]].copy()
            if cx[pi]:
                c1, c2 = crossover(a, b)
            else:
                c1, c2 = a, b
            if m1[pi]:
                c1 = mutate(c1)
            if m2[pi]:
                c2 = mutate(c2)
            new_pop.append(repair(c1))
            if len(new_pop) < pop_size:
                new_pop.append(repair(c2))
            pi += 1

        population = new_pop

    best_df = pd.DataFrame(best_rows)
    front = ["generation", "best_so_far_reach", "Reach", "Reach %",
             "Combination Number", "Skills", "no_improve_streak"]
    existing = [c for c in front if c in best_df.columns]
    return best_df[existing + [c for c in best_df.columns if c not in existing]]
