"""
TURF analysis primitives.

Contains the job-skill incidence matrix builder, the vectorised reach
computation used as the GA fitness function, the exhaustive optimum (for small r),
and the greedy TURF baseline.

Reach is the *unduplicated* number of OJAs covered by a skill bundle: an OJA
counts once if it contains at least one skill from the bundle.
"""
from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd


def build_incidence_matrix(turf_df: pd.DataFrame, skills: List[str]) -> np.ndarray:
    """Boolean matrix X (N_jobs x M); X[j, m] = True if OJA j demands skill m.

    ``turf_df`` holds 'yes'/'no' per skill column (the project's native format).
    """
    return (turf_df[skills] == "yes").to_numpy(dtype=np.bool_)


def vectorized_reach(X: np.ndarray, population: List[np.ndarray]) -> np.ndarray:
    """Reach for every chromosome at once via a single matrix multiply.

    X          : bool array (N_jobs, M)
    population : list of length-M int/bool vectors
    returns    : int32 array (pop_size,) of OJA counts covered

    ``covered = X @ Z.T`` has shape (N_jobs, P); an entry > 0 means that OJA is
    covered by that chromosome. Summing the boolean ``> 0`` over the OJA axis
    gives the unduplicated reach per chromosome.
    """
    Z = np.array(population, dtype=np.bool_)          # (P, M)
    covered = X @ Z.T                                  # (N_jobs, P)
    return (covered > 0).sum(axis=0).astype(np.int32)  # (P,)


def exhaustive_best(X: np.ndarray, r: int):
    """Evaluate every C(M, r) combination; return (best_reach, best_vector).

    Use only for small r — the number of combinations explodes quickly.
    """
    M = X.shape[1]
    population = []
    for combo in itertools.combinations(range(M), r):
        vec = np.zeros(M, dtype=np.int8)
        for i in combo:
            vec[i] = 1
        population.append(vec)
    reaches = vectorized_reach(X, population)
    best_idx = int(np.argmax(reaches))
    return float(reaches[best_idx]), population[best_idx]


def greedy_turf(
    X: np.ndarray,
    skill_ids: List[str],
    label_map: Dict[str, str],
    n_jobs: int,
) -> pd.DataFrame:
    """Classic greedy TURF: repeatedly add the skill with the largest marginal
    coverage. Deterministic and near-optimal for the (monotone submodular) reach
    objective, but not guaranteed globally optimal. Complexity O(M^2 * N).
    Returns one row per bundle size r = 1 .. M-1.
    """
    M = X.shape[1]
    covered = np.zeros(n_jobs, dtype=np.bool_)
    bundle: List[int] = []
    rows = []
    for step in range(1, M):
        best_marginal, best_idx = -1, -1
        for i in range(M):
            if i in bundle:
                continue
            new_cover = int((X[:, i] & ~covered).sum())
            if new_cover > best_marginal:
                best_marginal, best_idx = new_cover, i
        bundle.append(best_idx)
        covered |= X[:, best_idx].astype(bool)
        reach = int(covered.sum())
        labels = ", ".join(label_map.get(skill_ids[i], skill_ids[i]) for i in bundle)
        rows.append({
            "r": step,
            "reach": reach,
            "reach_pct": round(reach / n_jobs * 100, 2) if n_jobs else 0.0,
            "skills": labels,
        })
    return pd.DataFrame(rows)


def bundle_label(X_vec: np.ndarray, skill_ids: List[str], label_map: Dict[str, str]) -> str:
    """Human-readable label for a binary bundle vector."""
    return ", ".join(
        label_map.get(skill_ids[i], skill_ids[i])
        for i in range(len(skill_ids)) if int(X_vec[i]) == 1
    )
