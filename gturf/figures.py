"""
Figure generation for the G-TURF pipeline.

Every function takes the :class:`PipelineResults` (and, where relevant, the
statistics / sensitivity dataframes) and writes a PNG into the output figures
directory. All figures use a consistent palette and are sized to remain legible
at print resolution.
"""
from __future__ import annotations

import os
from math import comb
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from .config import GTurfConfig
from .pipeline import PipelineResults

PALETTE = ["#2E6FBF", "#E05C2A", "#2A9E6F", "#9E2A7A",
           "#C0392B", "#16A085", "#8E44AD", "#D35400"]


def _figpath(cfg: GTurfConfig, name: str) -> str:
    figures_dir = os.path.join(cfg.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return os.path.join(figures_dir, name)


def _occ_palette(occ_codes: List[str]) -> Dict[str, str]:
    return {occ: PALETTE[i % len(PALETTE)] for i, occ in enumerate(occ_codes)}


def fig_hcv_heatmap(results: PipelineResults, cfg: GTurfConfig) -> str:
    """Level-L HCV priorities across occupations (one column per occupation)."""
    sns.set_style("whitegrid")
    occ_codes = list(results.hcv.keys())
    top_skills: List[str] = []
    for occ in occ_codes:
        df = results.hcv[occ]
        top = (df[df["level"] == cfg.hcv_level]
               .sort_values("normalized priority", ascending=False)
               .head(cfg.top_m)["skill"].tolist())
        top_skills.extend(top)
    unique_skills = list(dict.fromkeys(top_skills))
    if not unique_skills:
        return ""
    data = pd.DataFrame(index=unique_skills, columns=occ_codes, dtype=float)
    for occ in occ_codes:
        l = results.hcv[occ][results.hcv[occ]["level"] == cfg.hcv_level].set_index("skill")
        for skill in unique_skills:
            data.loc[skill, occ] = float(l.loc[skill, "normalized priority"]) if skill in l.index else 0.0

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(occ_codes)),
                                    max(8, len(unique_skills) * 0.38)))
    sns.heatmap(data.astype(float), ax=ax, cmap="YlOrRd", linewidths=0.4,
                linecolor="white", annot=True, fmt=".3f", annot_kws={"size": 7},
                cbar_kws={"label": "Normalised priority"})
    ax.set_title(f"HCV level-{cfg.hcv_level} priorities ({cfg.pillar}, top {cfg.top_m})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Occupation"); ax.set_ylabel("Skill")
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    path = _figpath(cfg, "fig_hcv_heatmap.png")
    fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_reach_vs_r(results: PipelineResults, cfg: GTurfConfig,
                   stats_df: Optional[pd.DataFrame] = None) -> str:
    """Reach vs bundle size with 95% CI bands (if statistics supplied) or +/-std."""
    sns.set_style("whitegrid")
    occ_codes = list(results.summary.keys())
    cmap = _occ_palette(occ_codes)
    fig, ax = plt.subplots(figsize=(11, 5))
    for occ in occ_codes:
        s = results.summary[occ].set_index("r")
        rs = list(s.index)
        n_jobs = len(results.turf_df[occ])
        means = s["best_so_far_reach_mean"] / n_jobs * 100
        if stats_df is not None and not stats_df.empty:
            sub = stats_df[stats_df["occupation"] == occ].set_index("r")
            lo = [sub.loc[r, "ci95_low_pct"] if r in sub.index else means.loc[r] for r in rs]
            hi = [sub.loc[r, "ci95_high_pct"] if r in sub.index else means.loc[r] for r in rs]
        else:
            std = s["best_so_far_reach_std"] / n_jobs * 100
            lo, hi = means - std, means + std
        ax.plot(rs, means, marker="o", markersize=5, linewidth=2, color=cmap[occ], label=occ)
        ax.fill_between(rs, lo, hi, alpha=0.13, color=cmap[occ])
    ax.set_xlabel("Bundle size  r"); ax.set_ylabel("Reach (% of retained OJAs)")
    band = "95% CI" if stats_df is not None else "± std"
    ax.set_title(f"G-TURF optimal reach vs bundle size (M={cfg.top_m}, mean {band})",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(title="Occupation"); sns.despine()
    plt.tight_layout()
    path = _figpath(cfg, "fig_reach_vs_r.png")
    fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_gturf_vs_greedy(results: PipelineResults, cfg: GTurfConfig) -> str:
    """Per-occupation panels: G-TURF vs greedy TURF reach curves."""
    sns.set_style("whitegrid")
    occ_codes = list(results.summary.keys())
    cmap = _occ_palette(occ_codes)
    ncol = 2; nrow = int(np.ceil(len(occ_codes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 5 * nrow), squeeze=False)
    for i, occ in enumerate(occ_codes):
        ax = axes.flatten()[i]
        s = results.summary[occ].set_index("r")
        g = results.greedy[occ].set_index("r")
        rs = [r for r in s.index if r in g.index]
        n_jobs = len(results.turf_df[occ])
        gturf = [s.loc[r, "best_so_far_reach_mean"] / n_jobs * 100 for r in rs]
        greedy = [g.loc[r, "reach_pct"] for r in rs]
        ax.plot(rs, gturf, marker="o", markersize=4, linewidth=2.2, color=cmap[occ],
                label="G-TURF (GA)")
        ax.plot(rs, greedy, marker="s", markersize=4, linewidth=2, linestyle="--",
                color=cmap[occ], alpha=0.7, label="Greedy TURF")
        ax.set_title(occ, fontsize=11, fontweight="bold")
        ax.set_xlabel("Bundle size  r"); ax.set_ylabel("Reach (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(fontsize=8, loc="lower right"); sns.despine(ax=ax)
    for j in range(len(occ_codes), nrow * ncol):
        axes.flatten()[j].axis("off")
    fig.suptitle(f"G-TURF vs Greedy TURF (M={cfg.top_m})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = _figpath(cfg, "fig_gturf_vs_greedy.png")
    fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_hcv_vs_gturf(results: PipelineResults, cfg: GTurfConfig) -> str:
    """Per-occupation panels: HCV top-r naive baseline vs G-TURF optimal."""
    sns.set_style("whitegrid")
    occ_codes = list(results.hcv_vs_turf.keys())
    cmap = _occ_palette(occ_codes)
    ncol = 2; nrow = int(np.ceil(len(occ_codes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 5 * nrow), squeeze=False)
    for i, occ in enumerate(occ_codes):
        ax = axes.flatten()[i]
        df = results.hcv_vs_turf[occ].set_index("r")
        rs = list(df.index)
        ax.plot(rs, df["hcv_top_r_reach_%"], marker="s", markersize=5, linewidth=2,
                color="#888888", linestyle="--", label="HCV top-r (naive)")
        ax.plot(rs, df["gturf_reach_%"], marker="o", markersize=5, linewidth=2,
                color=cmap[occ], label="G-TURF optimal")
        ax.fill_between(rs, df["hcv_top_r_reach_%"], df["gturf_reach_%"],
                        alpha=0.18, color=cmap[occ], label="G-TURF gain")
        ax.set_title(occ, fontsize=11, fontweight="bold")
        ax.set_xlabel("Bundle size  r"); ax.set_ylabel("Reach (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(fontsize=8); sns.despine(ax=ax)
    for j in range(len(occ_codes), nrow * ncol):
        axes.flatten()[j].axis("off")
    fig.suptitle(f"HCV top-r baseline vs G-TURF optimal (M={cfg.top_m})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = _figpath(cfg, "fig_hcv_vs_gturf.png")
    fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_bundle_composition(results: PipelineResults, cfg: GTurfConfig) -> str:
    """Per-occupation panels showing which skills are in the optimal bundle at
    each r. Skills are shown as short codes S1..SM with a printed code->name
    legend below, to stay legible."""
    sns.set_style("white")
    occ_codes = list(results.summary.keys())
    ncol = 2; nrow = int(np.ceil(len(occ_codes) / ncol))
    fig = plt.figure(figsize=(20, 7.5 * nrow))
    gs = fig.add_gridspec(nrow, ncol, top=0.93, bottom=0.34, hspace=0.30, wspace=0.12,
                          left=0.05, right=0.985)
    blue, cream = "#2E6FBF", "#EDE7D9"
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([cream, blue])

    label_lines = {}
    for idx, occ in enumerate(occ_codes):
        ax = fig.add_subplot(gs[idx // ncol, idx % ncol])
        candidate_ids = results.candidate_skills[occ]
        rs = list(results.summary[occ]["r"])
        M = np.zeros((len(rs), len(candidate_ids)), dtype=int)
        for row_i, r in enumerate(rs):
            histories = results.run_histories[occ][r]
            best = histories[int(np.argmax([h["best_so_far_reach"].iloc[-1] for h in histories]))]
            last = best.iloc[-1]
            for col_i, sid in enumerate(candidate_ids):
                if sid in last and int(last[sid]) == 1:
                    M[row_i, col_i] = 1
        ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, len(candidate_ids), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rs), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(len(candidate_ids)))
        ax.set_xticklabels([f"S{i+1}" for i in range(len(candidate_ids))],
                           fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(rs))); ax.set_yticklabels([f"r={r}" for r in rs], fontsize=10)
        ax.set_title(occ, fontsize=14, fontweight="bold")
        ax.set_xlabel("Skill code"); ax.set_ylabel("Bundle size r")
        label_lines[occ] = [results.label_map.get(s, s) for s in candidate_ids]

    fig.legend(handles=[mpatches.Patch(color=blue, label="In optimal bundle"),
                        mpatches.Patch(color=cream, label="Not included")],
               loc="upper center", ncol=2, fontsize=13,
               bbox_to_anchor=(0.5, 0.965), frameon=False)
    fig.suptitle(f"G-TURF optimal bundle composition ({cfg.pillar}, M={cfg.top_m})",
                 fontsize=16, fontweight="bold", y=0.99)
    # code -> name mapping below
    col_x = np.linspace(0.06, 0.78, ncol).tolist()
    fig.text(0.5, 0.30, "Skill code mapping (per occupation, HCV-priority order)",
             ha="center", fontsize=14, fontweight="bold")
    for ci, occ in enumerate(occ_codes):
        x = col_x[ci % ncol]
        y = 0.275 - 0.13 * (ci // ncol)
        body = "\n".join(f"S{i+1}: {lbl[:30] + ('…' if len(lbl) > 30 else '')}"
                         for i, lbl in enumerate(label_lines[occ]))
        fig.text(x, y, f"{occ}\n{body}", ha="left", va="top", fontsize=8.5, linespacing=1.35)
    path = _figpath(cfg, "fig_bundle_composition.png")
    fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_computational_scaling(results: PipelineResults, cfg: GTurfConfig) -> str:
    """Search-space size, runtime, speedup, and theoretical M=20/50/100 scaling."""
    sns.set_style("whitegrid")
    occ = list(results.summary.keys())[0]
    s = results.summary[occ].set_index("r")
    r_all = list(s.index)
    if cfg.exhaustive_threshold + 1 not in s.index and 4 in s.index:
        calib_r = 4
    else:
        calib_r = min(cfg.exhaustive_threshold, max(r_all))
    space = [comb(cfg.top_m, r) for r in r_all]
    ga_times = [float(s.loc[r, "elapsed_s"]) for r in r_all]
    cost_per_combo = float(s.loc[calib_r, "elapsed_s"]) / comb(cfg.top_m, calib_r)
    exh_times = [float(s.loc[r, "elapsed_s"]) if r <= cfg.exhaustive_threshold
                 else comb(cfg.top_m, r) * cost_per_combo for r in r_all]
    speedups = [e / g if g > 0 else 0 for e, g in zip(exh_times, ga_times)]
    ga_flat = float(np.median(ga_times))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax = axes[0, 0]
    ax.bar(r_all, space, color="#2E6FBF", alpha=0.82)
    ax.set_yscale("log"); ax.set_title(f"Search space C({cfg.top_m}, r)", fontweight="bold")
    ax.set_xlabel("r"); ax.set_ylabel("combinations"); sns.despine(ax=ax)

    ax = axes[0, 1]
    ax.plot(r_all, exh_times, marker="s", color="#C0392B", label="Exhaustive (extrap.)")
    ax.plot(r_all, ga_times, marker="o", color="#2E6FBF", label="G-TURF")
    ax.set_yscale("log"); ax.set_title("Runtime", fontweight="bold")
    ax.set_xlabel("r"); ax.set_ylabel("seconds"); ax.legend(); sns.despine(ax=ax)

    ax = axes[1, 0]
    colors = ["#2A9E6F" if v >= 1 else "#E05C2A" for v in speedups]
    ax.bar(r_all, speedups, color=colors, alpha=0.85)
    ax.axhline(1, color="#333", linestyle="--", linewidth=1)
    ax.set_title("Speedup (exhaustive / GA)", fontweight="bold")
    ax.set_xlabel("r"); ax.set_ylabel("x"); sns.despine(ax=ax)

    ax = axes[1, 1]
    SEC_YR = 365.25 * 24 * 3600
    for Mx, col, ls in zip([20, 50, 100], ["#2E6FBF", "#E07A2A", "#8E2AC0"], ["-", "--", ":"]):
        rr = list(range(2, min(Mx, 50)))
        ax.plot(rr, [comb(Mx, r) * cost_per_combo for r in rr], color=col, linestyle=ls,
                linewidth=2, label=f"Exhaustive M={Mx}")
    ax.axhline(ga_flat, color="#555", linewidth=2.5, label=f"G-TURF (~{ga_flat:.0f}s)")
    for y, lbl in [(3600, "1 hour"), (86400, "1 day"), (SEC_YR, "1 year")]:
        ax.axhline(y, color="#CCC", linestyle=":", linewidth=1)
    ax.set_yscale("log"); ax.set_title("Theoretical scaling M=20/50/100", fontweight="bold")
    ax.set_xlabel("r"); ax.set_ylabel("seconds"); ax.legend(fontsize=8); sns.despine(ax=ax)

    fig.suptitle(f"Computational scaling ({occ})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = _figpath(cfg, "fig_computational_scaling.png")
    fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_sensitivity(sens_df: pd.DataFrame, cfg: GTurfConfig) -> str:
    """Grid of line plots: mean reach % vs each swept parameter, per occupation."""
    if sens_df.empty:
        return ""
    sns.set_style("whitegrid")
    params = sorted(sens_df["parameter"].unique())
    occ_codes = sorted(sens_df["occupation"].unique())
    cmap = _occ_palette(occ_codes)
    ncol = 2; nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 4.5 * nrow), squeeze=False)
    for i, param in enumerate(params):
        ax = axes.flatten()[i]
        for occ in occ_codes:
            sub = sens_df[(sens_df["parameter"] == param) & (sens_df["occupation"] == occ)]
            sub = sub.sort_values("value")
            ax.plot(sub["value"], sub["mean_pct"], marker="o", color=cmap[occ], label=occ)
        ax.set_title(param, fontsize=11, fontweight="bold")
        ax.set_xlabel(param); ax.set_ylabel("mean reach (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax.legend(fontsize=8); sns.despine(ax=ax)
    for j in range(len(params), nrow * ncol):
        axes.flatten()[j].axis("off")
    fig.suptitle("GA hyperparameter sensitivity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = _figpath(cfg, "fig_sensitivity.png")
    fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight"); plt.close(fig)
    return path


def generate_all_figures(results: PipelineResults, cfg: GTurfConfig,
                         stats_df: Optional[pd.DataFrame] = None) -> List[str]:
    """Produce the standard figure set; returns the list of written paths."""
    paths = []
    for fn in (fig_hcv_heatmap, fig_gturf_vs_greedy, fig_hcv_vs_gturf,
               fig_bundle_composition, fig_computational_scaling):
        try:
            p = fn(results, cfg)
            if p:
                paths.append(p)
        except Exception as exc:  # keep going if one figure fails
            print(f"  [warn] {fn.__name__} failed: {exc}")
    try:
        p = fig_reach_vs_r(results, cfg, stats_df)
        if p:
            paths.append(p)
    except Exception as exc:
        print(f"  [warn] fig_reach_vs_r failed: {exc}")
    return paths
