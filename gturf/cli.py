"""
Command-line interfaces for G-TURF.

Three entry points are exposed:
  - :func:`pipeline_main`     (console script ``gturf-run``)
  - :func:`sensitivity_main`  (console script ``gturf-sensitivity``)
  - :func:`statistics_main`   (console script ``gturf-statistics``)

The thin wrappers under ``scripts/`` simply import and call these, so the same
code path is used whether the package is pip-installed or run from a checkout.
"""
from __future__ import annotations

import argparse
import os

from .config import GTurfConfig
from .pipeline import run_pipeline
from .statistics import compute_statistics, summarise_statistics
from .sensitivity import run_sensitivity, summarise_sensitivity, DEFAULT_GRIDS
from .figures import generate_all_figures, fig_sensitivity
from . import io_utils


# ── shared helpers ────────────────────────────────────────────────────────────

def _maybe_synthetic(args):
    if getattr(args, "synthetic", False):
        print("Generating synthetic dataset...")
        sample_dir = os.path.join(args.output_dir, "synthetic_data")
        oja, esco = io_utils.generate_synthetic_dataset(sample_dir)
        args.oja_path, args.esco_mapping_path = oja, esco
        print(f"  OJA file  : {oja}\n  ESCO file : {esco}")


def _occupations(args):
    if not getattr(args, "occupations", None):
        return None
    default = GTurfConfig().occupations
    return {c: default.get(c, f"http://data.europa.eu/esco/isco/{c}")
            for c in args.occupations}


# ── gturf-run ─────────────────────────────────────────────────────────────────

def _pipeline_parser():
    p = argparse.ArgumentParser(
        prog="gturf-run",
        description="Run the G-TURF pipeline (HCV -> TURF/GA) end to end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--oja", dest="oja_path", default="jobs_software_engineer.xlsx")
    p.add_argument("--esco-mapping", dest="esco_mapping_path", default="new_ESCO_mapping.xlsx")
    p.add_argument("--output-dir", default="gturf_output")
    p.add_argument("--pillar", default="knowledge", choices=["skills", "knowledge", "traversal"])
    p.add_argument("--occupations", nargs="*", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--top-m", type=int, default=20)
    p.add_argument("--hcv-level", type=int, default=4)
    p.add_argument("--crossover-rate", type=float, default=0.8)
    p.add_argument("--mutation-rate", type=float, default=0.25)
    p.add_argument("--elitism", type=int, default=2)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--min-delta", type=int, default=1)
    p.add_argument("--init-frac", type=float, default=1.0 / 3.0)
    p.add_argument("--max-pop", type=int, default=8000)
    p.add_argument("--runs-per-r", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=100)
    p.add_argument("--exhaustive-threshold", type=int, default=4)
    p.add_argument("--r-min", type=int, default=2)
    p.add_argument("--r-max", type=int, default=None)
    p.add_argument("--no-monotonicity", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-excel", action="store_true")
    p.add_argument("--no-stats", action="store_true")
    p.add_argument("--figure-dpi", type=int, default=200)
    return p


def pipeline_main(argv=None):
    args = _pipeline_parser().parse_args(argv)
    _maybe_synthetic(args)

    kwargs = dict(
        oja_path=args.oja_path, esco_mapping_path=args.esco_mapping_path,
        output_dir=args.output_dir, pillar=args.pillar, top_m=args.top_m,
        hcv_level=args.hcv_level, crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate, elitism=args.elitism,
        generations=args.generations, early_stop_patience=args.early_stop_patience,
        min_delta=args.min_delta, init_frac=args.init_frac, max_pop=args.max_pop,
        runs_per_r=args.runs_per_r, base_seed=args.base_seed,
        exhaustive_threshold=args.exhaustive_threshold, r_min=args.r_min,
        r_max=args.r_max, enforce_monotonicity=not args.no_monotonicity,
        save_figures=not args.no_figures, save_excel=not args.no_excel,
        figure_dpi=args.figure_dpi,
    )
    occ = _occupations(args)
    if occ:
        kwargs["occupations"] = occ
    cfg = GTurfConfig(**kwargs)

    print(f"\nRunning G-TURF | pillar={cfg.pillar} | M={cfg.top_m} | "
          f"r={cfg.r_min}..{cfg.r_max} | runs/r={cfg.runs_per_r}")
    results = run_pipeline(cfg, verbose=True)

    stats_df = None
    if not args.no_stats:
        print("\nComputing statistics (CI, Wilcoxon, TOST)...")
        stats_df = compute_statistics(results)
        if cfg.save_excel and not stats_df.empty:
            stats_df.to_excel(os.path.join(cfg.output_dir, "statistics.xlsx"), index=False)
        summary = summarise_statistics(stats_df)
        if summary:
            print("  " + " | ".join(f"{k}={v}" for k, v in summary.items()))

    if not args.no_figures:
        print("\nGenerating figures...")
        for p in generate_all_figures(results, cfg, stats_df):
            print(f"  {p}")

    print(f"\nDone. Outputs in: {cfg.output_dir}/")


# ── gturf-sensitivity ─────────────────────────────────────────────────────────

def _sensitivity_parser():
    p = argparse.ArgumentParser(
        prog="gturf-sensitivity",
        description="GA hyperparameter sensitivity analysis for G-TURF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--oja", dest="oja_path", default="jobs_software_engineer.xlsx")
    p.add_argument("--esco-mapping", dest="esco_mapping_path", default="new_ESCO_mapping.xlsx")
    p.add_argument("--output-dir", default="gturf_sensitivity")
    p.add_argument("--pillar", default="knowledge", choices=["skills", "knowledge", "traversal"])
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--top-m", type=int, default=20)
    p.add_argument("--r", type=int, default=10)
    p.add_argument("--parameters", nargs="*", default=list(DEFAULT_GRIDS.keys()))
    p.add_argument("--runs-per-setting", type=int, default=3)
    return p


def sensitivity_main(argv=None):
    args = _sensitivity_parser().parse_args(argv)
    _maybe_synthetic(args)

    cfg = GTurfConfig(
        oja_path=args.oja_path, esco_mapping_path=args.esco_mapping_path,
        output_dir=args.output_dir, pillar=args.pillar, top_m=args.top_m,
        save_figures=False, save_excel=False,
    )
    print("Running base pipeline to obtain candidate sets...")
    results = run_pipeline(cfg, verbose=False)

    print(f"\nSweeping {args.parameters} at r={args.r}...")
    sens_df = run_sensitivity(results, cfg, r=args.r, parameters=args.parameters,
                              runs_per_setting=args.runs_per_setting, verbose=True)
    os.makedirs(args.output_dir, exist_ok=True)
    sens_df.to_excel(os.path.join(args.output_dir, "sensitivity_results.xlsx"), index=False)
    summary = summarise_sensitivity(sens_df)
    summary.to_excel(os.path.join(args.output_dir, "sensitivity_summary.xlsx"), index=False)
    fig_path = fig_sensitivity(sens_df, cfg)

    print("\nReach range per parameter (smaller = more robust):")
    print(summary.to_string(index=False))
    if fig_path:
        print(f"\nFigure: {fig_path}")
    print(f"Done. Outputs in: {args.output_dir}/")


# ── gturf-statistics ──────────────────────────────────────────────────────────

def _statistics_parser():
    p = argparse.ArgumentParser(
        prog="gturf-statistics",
        description="Statistical analysis for G-TURF (CI / Wilcoxon / TOST).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--oja", dest="oja_path", default="jobs_software_engineer.xlsx")
    p.add_argument("--esco-mapping", dest="esco_mapping_path", default="new_ESCO_mapping.xlsx")
    p.add_argument("--output-dir", default="gturf_statistics")
    p.add_argument("--pillar", default="knowledge", choices=["skills", "knowledge", "traversal"])
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--top-m", type=int, default=20)
    p.add_argument("--runs-per-r", type=int, default=5)
    p.add_argument("--tost-margin", type=float, default=0.5)
    p.add_argument("--confidence", type=float, default=0.95)
    return p


def statistics_main(argv=None):
    args = _statistics_parser().parse_args(argv)
    _maybe_synthetic(args)

    cfg = GTurfConfig(
        oja_path=args.oja_path, esco_mapping_path=args.esco_mapping_path,
        output_dir=args.output_dir, pillar=args.pillar, top_m=args.top_m,
        runs_per_r=args.runs_per_r, save_figures=False, save_excel=True,
    )
    print("Running pipeline...")
    results = run_pipeline(cfg, verbose=False)

    print(f"Computing statistics (TOST margin = {args.tost_margin} pp)...")
    stats_df = compute_statistics(results, tost_margin_pp=args.tost_margin,
                                  confidence=args.confidence)
    os.makedirs(args.output_dir, exist_ok=True)
    stats_df.to_excel(os.path.join(args.output_dir, "statistics_full.xlsx"), index=False)

    summary = summarise_statistics(stats_df)
    print("\nHeadline statistics:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull table: {os.path.join(args.output_dir, 'statistics_full.xlsx')}")
