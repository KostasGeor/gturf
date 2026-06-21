"""
G-TURF pipeline orchestrator.

Ties the stages together for each occupation:

  1. Load OJA skill lists + ESCO mapping.
  2. Run HCV to obtain occupation-specific priorities; keep the top-M level-L
     skills as the candidate set.
  3. Build the job-skill incidence matrix over the candidate set.
  4. For each bundle size r: exhaustive search (small r) or GA (larger r),
     repeated ``runs_per_r`` times.
  5. Compute the greedy TURF baseline and the HCV-top-r naive baseline.
  6. Enforce monotonicity of the reach-vs-r curve (optional).

Results are returned in a :class:`PipelineResults` container and (optionally)
written to Excel. Figures are produced separately by :mod:`gturf.figures`.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import GTurfConfig
from .hcv import run_hcv, top_m_candidate_skills
from .turf import (
    build_incidence_matrix, vectorized_reach, exhaustive_best, greedy_turf,
)
from .ga import run_ga
from . import io_utils


@dataclass
class PipelineResults:
    """All artefacts produced for the full set of occupations."""
    config: GTurfConfig
    hcv: Dict[str, pd.DataFrame] = field(default_factory=dict)
    candidate_skills: Dict[str, List[str]] = field(default_factory=dict)
    turf_df: Dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: Dict[str, pd.DataFrame] = field(default_factory=dict)
    run_histories: Dict[str, Dict[int, List[pd.DataFrame]]] = field(default_factory=dict)
    greedy: Dict[str, pd.DataFrame] = field(default_factory=dict)
    hcv_vs_turf: Dict[str, pd.DataFrame] = field(default_factory=dict)
    label_map: Dict[str, str] = field(default_factory=dict)


def _build_turf_df(skills_list: List[List[str]], candidate_ids: List[str]) -> pd.DataFrame:
    """yes/no matrix over the candidate skills, keeping only OJAs with >=1 hit."""
    df = pd.DataFrame([
        {skill: ("yes" if skill in set(sl) else "no") for skill in candidate_ids}
        for sl in skills_list
    ])
    if df.empty:
        return df
    keep = df.eq("yes").any(axis=1)
    return df.loc[keep].reset_index(drop=True)


def run_pipeline(cfg: GTurfConfig, verbose: bool = True) -> PipelineResults:
    """Execute the full G-TURF pipeline for every occupation in ``cfg``."""
    skills_df, label_map = io_utils.load_esco_mapping(cfg.esco_mapping_path)
    oja_by_occ = io_utils.load_oja_skill_lists(cfg.oja_path, cfg.occupations)

    results = PipelineResults(config=cfg, label_map=label_map)

    if cfg.save_excel or cfg.save_figures:
        io_utils.ensure_dirs(cfg.output_dir)
        cfg.to_json(os.path.join(cfg.output_dir, "run_config.json"))

    for occ_code in cfg.occupations:
        if verbose:
            print(f"\n{'=' * 60}\n  Occupation: {occ_code}\n{'=' * 60}")
        skills_list = oja_by_occ.get(occ_code, [])
        if verbose:
            print(f"OJAs loaded: {len(skills_list)}")
        if not skills_list:
            if verbose:
                print(f"  [skip] no OJAs for {occ_code}")
            continue

        # ── HCV ──────────────────────────────────────────────────────────────
        hcv_df = run_hcv(
            skills_list, skills_df, label_map,
            cfg.levels_column, cfg.ancestors_column,
        )
        results.hcv[occ_code] = hcv_df
        candidate_ids = top_m_candidate_skills(hcv_df, cfg.hcv_level, cfg.top_m)
        # Fall back to the deepest populated level if the requested level is empty
        # (e.g. taxonomies shallower than 4 levels, or sparse synthetic data).
        if not candidate_ids and not hcv_df.empty:
            deepest = int(hcv_df["level"].max())
            if deepest != cfg.hcv_level:
                if verbose:
                    print(f"  [warn] level {cfg.hcv_level} empty; "
                          f"using deepest populated level {deepest} instead.")
                candidate_ids = top_m_candidate_skills(hcv_df, deepest, cfg.top_m)
        results.candidate_skills[occ_code] = candidate_ids
        if not candidate_ids:
            if verbose:
                print(f"  [skip] no candidate skills produced for {occ_code}.")
            continue
        if verbose:
            print(f"Top-{cfg.top_m} candidate skills: "
                  f"{[label_map.get(s, s) for s in candidate_ids]}")
        if len(candidate_ids) < cfg.r_max + 1:
            if verbose:
                print(f"  [warn] only {len(candidate_ids)} candidate skills; "
                      f"reducing r_max accordingly.")

        # ── incidence matrix ──────────────────────────────────────────────────
        turf_df = _build_turf_df(skills_list, candidate_ids)
        results.turf_df[occ_code] = turf_df
        n_jobs = len(turf_df)
        if verbose:
            print(f"OJAs retained (>=1 candidate skill): {n_jobs}")
        if n_jobs == 0:
            continue
        X = build_incidence_matrix(turf_df, candidate_ids)

        occ_dir = io_utils.occupation_dir(cfg.output_dir, occ_code) if cfg.save_excel else None
        if cfg.save_excel:
            hcv_df.to_excel(os.path.join(occ_dir, f"HCV_results_{occ_code}.xlsx"), index=False)

        # ── per-r search ───────────────────────────────────────────────────────
        max_r = min(cfg.r_max, len(candidate_ids) - 1)
        if max_r < cfg.r_min:
            if verbose:
                print(f"  [skip] candidate set too small ({len(candidate_ids)}) "
                      f"for r_min={cfg.r_min}.")
            continue
        r_range = list(range(cfg.r_min, max_r + 1))
        summary_rows, run_histories = [], {}

        for r in r_range:
            t0 = time.time()
            if r <= cfg.exhaustive_threshold:
                best_reach, best_vec = exhaustive_best(X, r)
                elapsed = time.time() - t0
                row = _synthetic_history_row(best_reach, best_vec, candidate_ids, r, n_jobs, label_map)
                syn_df = pd.DataFrame([row])
                run_histories[r] = [syn_df] * cfg.runs_per_r
                best_runs = [best_reach] * cfg.runs_per_r
                gens_used = [1] * cfg.runs_per_r
                method = "exhaustive"
                if cfg.save_excel:
                    syn_df.to_excel(os.path.join(occ_dir, f"EXHAUSTIVE_r{r}.xlsx"), index=False)
                if verbose:
                    print(f"  r={r:2d} | EXHAUSTIVE | reach={best_reach:.0f} ({elapsed:.1f}s)")
            else:
                best_runs, gens_used, run_histories[r] = [], [], []
                for run_i in range(cfg.runs_per_r):
                    seed = cfg.base_seed + r * 1000 + run_i
                    bdf = run_ga(
                        X, candidate_ids, label_map, r,
                        generations=cfg.generations,
                        crossover_rate=cfg.crossover_rate,
                        mutation_rate=cfg.mutation_rate,
                        elitism=cfg.elitism,
                        seed=seed,
                        early_stop_patience=cfg.early_stop_patience,
                        min_delta=cfg.min_delta,
                        init_frac=cfg.init_frac,
                        max_pop=cfg.max_pop,
                        n_jobs=n_jobs,
                    )
                    run_histories[r].append(bdf)
                    best_runs.append(float(bdf["best_so_far_reach"].iloc[-1]))
                    gens_used.append(int(bdf["generation"].iloc[-1]))
                    if cfg.save_excel:
                        bdf.to_excel(
                            os.path.join(occ_dir, f"GA_r{r}_run{run_i + 1}_seed{seed}.xlsx"),
                            index=False,
                        )
                method = "GA"
                elapsed = time.time() - t0
                if verbose:
                    print(f"  r={r:2d} | GA | best reach={max(best_runs):.0f} "
                          f"| mean gens={np.mean(gens_used):.1f} ({elapsed:.1f}s)")

            best_label = run_histories[r][int(np.argmax(best_runs))]["Skills"].iloc[-1]
            summary_rows.append({
                "r": r,
                "method": method,
                "runs": 1 if method == "exhaustive" else cfg.runs_per_r,
                "best_so_far_reach_mean": float(np.mean(best_runs)),
                "best_so_far_reach_std": 0.0 if method == "exhaustive" else float(np.std(best_runs)),
                "generations_used_mean": float(np.mean(gens_used)),
                "best_skills": best_label,
                "elapsed_s": elapsed,
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("r").reset_index(drop=True)

        # ── monotonicity ────────────────────────────────────────────────────────
        if cfg.enforce_monotonicity and not summary_df.empty:
            raw = summary_df["best_so_far_reach_mean"].values.copy()
            mono = np.maximum.accumulate(raw)
            if not np.array_equal(mono, raw):
                n_fixed = int((mono != raw).sum())
                if verbose:
                    print(f"  Monotonicity fix applied to {n_fixed} r-value(s)")
                summary_df["best_so_far_reach_mean"] = mono
                summary_df.loc[mono != raw, "best_so_far_reach_std"] = 0.0

        if cfg.save_excel:
            summary_df.to_excel(os.path.join(occ_dir, f"summary_by_r_{occ_code}.xlsx"), index=False)
        results.summary[occ_code] = summary_df
        results.run_histories[occ_code] = run_histories

        # ── greedy baseline ─────────────────────────────────────────────────────
        greedy_df = greedy_turf(X, candidate_ids, label_map, n_jobs)
        results.greedy[occ_code] = greedy_df
        if cfg.save_excel:
            greedy_df.to_excel(os.path.join(occ_dir, f"greedy_turf_{occ_code}.xlsx"), index=False)

        # ── HCV-top-r naive baseline vs G-TURF ──────────────────────────────────
        results.hcv_vs_turf[occ_code] = _hcv_vs_turf(
            candidate_ids, turf_df, run_histories, r_range, n_jobs, label_map
        )
        if cfg.save_excel:
            results.hcv_vs_turf[occ_code].to_excel(
                os.path.join(occ_dir, f"hcv_vs_turf_comparison_{occ_code}.xlsx"), index=False
            )

    return results


def _synthetic_history_row(best_reach, best_vec, candidate_ids, r, n_jobs, label_map):
    from .turf import bundle_label
    row = {
        "generation": 1,
        "best_so_far_reach": best_reach,
        "Reach": best_reach,
        "Reach %": round(best_reach / n_jobs * 100, 2) if n_jobs else 0.0,
        "Combination Number": r,
        "Skills": bundle_label(best_vec, candidate_ids, label_map),
        "no_improve_streak": 0,
    }
    for i, sid in enumerate(candidate_ids):
        row[sid] = int(best_vec[i])
    return row


def _hcv_vs_turf(candidate_ids, turf_df, run_histories, r_range, n_jobs, label_map):
    rows = []
    for r in r_range:
        hcv_top_r = candidate_ids[:r]
        hcv_reach = int(turf_df[hcv_top_r].eq("yes").any(axis=1).sum())
        hcv_pct = round(hcv_reach / n_jobs * 100, 2)
        best_idx = int(np.argmax([df["best_so_far_reach"].iloc[-1] for df in run_histories[r]]))
        best_df = run_histories[r][best_idx]
        ga_reach = int(best_df["best_so_far_reach"].iloc[-1])
        ga_pct = round(ga_reach / n_jobs * 100, 2)
        last = best_df.iloc[-1]
        ga_set = set(s for s in candidate_ids if s in last and int(last[s]) == 1)
        rows.append({
            "r": r,
            "hcv_top_r_reach_%": hcv_pct,
            "gturf_reach_%": ga_pct,
            "gain_%": round(ga_pct - hcv_pct, 2),
            "bundles_differ": ga_set != set(hcv_top_r),
            "hcv_skills": ", ".join(label_map.get(s, s) for s in hcv_top_r),
            "gturf_skills": best_df["Skills"].iloc[-1],
        })
    return pd.DataFrame(rows)
