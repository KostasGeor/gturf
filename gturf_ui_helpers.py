"""
Helper functions for the G-TURF web UI's value-added features.

These are pure functions over a ``PipelineResults`` object so they can be unit
tested and reused. They power:
  - the plain-language results interpreter,
  - the reach-target decision tool (both directions),
  - the interactive bundle explorer,
  - the side-by-side occupation comparison.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


OCC_NAMES = {
    "C2511": "Systems Analysts",
    "C2512": "Software Developers",
    "C2513": "Web & Multimedia Developers",
    "C2514": "Applications Programmers",
}


def occ_display(code: str) -> str:
    name = OCC_NAMES.get(code)
    return f"{code} — {name}" if name else code


def reach_curve(results, occ: str) -> pd.DataFrame:
    """Return a tidy (r, reach_pct, skills) table for one occupation."""
    s = results.summary[occ]
    n_jobs = max(1, len(results.turf_df[occ]))
    out = pd.DataFrame({
        "r": s["r"].astype(int),
        "reach_pct": (s["best_so_far_reach_mean"] / n_jobs * 100).round(1),
        "skills": s["best_skills"],
    })
    return out.reset_index(drop=True)


# ── 1. Plain-language interpreter ─────────────────────────────────────────────

def interpret_results(results, knee_gain_threshold: float = 3.0) -> str:
    """Produce a plain-language Markdown summary of what the run means.

    For each occupation it reports the reach at a useful bundle size and finds
    the "knee" — the point past which adding skills yields < ``knee_gain_threshold``
    percentage points of extra reach.
    """
    if not results.summary:
        return "No results to interpret yet."

    lines = ["## What these results mean\n"]
    for occ in results.summary:
        curve = reach_curve(results, occ)
        if curve.empty:
            continue
        rs = curve["r"].tolist()
        reach = curve["reach_pct"].tolist()

        # knee: first r where the marginal gain to the next r drops below threshold
        knee_r = rs[-1]
        for i in range(len(rs) - 1):
            if reach[i + 1] - reach[i] < knee_gain_threshold:
                knee_r = rs[i]
                break
        knee_reach = curve.loc[curve["r"] == knee_r, "reach_pct"].iloc[0]
        max_reach = reach[-1]
        max_r = rs[-1]

        name = occ_display(occ)
        msg = (f"**{name}.** A bundle of **{knee_r} skills** already reaches "
               f"**{knee_reach:.0f}%** of job postings. ")
        if knee_r < max_r:
            extra = max_reach - knee_reach
            msg += (f"Going all the way to {max_r} skills adds only "
                    f"{extra:.0f} more percentage points ({max_reach:.0f}% total) — "
                    f"so the first {knee_r} skills carry most of the value.")
        else:
            msg += f"Reach keeps climbing across the whole range, up to {max_reach:.0f}% at r={max_r}."
        lines.append("- " + msg)

    lines.append(
        "\n*The “knee” is where adding another skill stops paying off "
        "(< {:.0f} pp extra reach). It's a practical guide for how many skills "
        "to prioritise under limited teaching or certification capacity.*"
        .format(knee_gain_threshold)
    )
    return "\n".join(lines)


# ── 2. Reach-target decision tool ─────────────────────────────────────────────

def skills_for_target(results, occ: str, target_pct: float) -> Tuple[Optional[int], Optional[float]]:
    """Smallest bundle size reaching >= target_pct. Returns (r, reach_pct) or (None, None)."""
    curve = reach_curve(results, occ)
    hit = curve[curve["reach_pct"] >= target_pct]
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    return int(row["r"]), float(row["reach_pct"])


def reach_for_budget(results, occ: str, budget: int) -> Tuple[Optional[int], Optional[float]]:
    """Reach achievable with exactly `budget` skills. Returns (r_used, reach_pct)."""
    curve = reach_curve(results, occ)
    avail = curve[curve["r"] <= budget]
    if avail.empty:
        return None, None
    row = avail.iloc[-1]
    return int(row["r"]), float(row["reach_pct"])


def decision_tool(results, mode: str, occ: str, value: float) -> str:
    """Markdown answer for the reach-target tool, in either direction."""
    if not results.summary or occ not in results.summary:
        return "Run the pipeline first, then pick an occupation."
    name = occ_display(occ)
    if mode == "budget":  # I can teach N skills -> what coverage?
        r, pct = reach_for_budget(results, occ, int(value))
        if r is None:
            return f"No bundle that small was evaluated for {name}."
        skills = reach_curve(results, occ)
        bundle = skills.loc[skills["r"] == r, "skills"].iloc[0]
        return (f"### {name}\n"
                f"With a budget of **{int(value)} skills**, the best bundle of "
                f"**{r}** reaches **{pct:.0f}%** of job postings.\n\n"
                f"**Recommended skills:** {bundle}")
    else:  # target coverage -> how many skills?
        r, pct = skills_for_target(results, occ, float(value))
        if r is None:
            max_pct = reach_curve(results, occ)["reach_pct"].max()
            return (f"### {name}\n"
                    f"Even the largest evaluated bundle reaches only "
                    f"**{max_pct:.0f}%**, short of your {value:.0f}% target. "
                    f"Try a larger r-max when running the pipeline.")
        skills = reach_curve(results, occ)
        bundle = skills.loc[skills["r"] == r, "skills"].iloc[0]
        return (f"### {name}\n"
                f"To reach **{value:.0f}%** coverage you need **{r} skills** "
                f"(achieving {pct:.0f}%).\n\n"
                f"**Recommended skills:** {bundle}")


# ── 3. Bundle explorer ────────────────────────────────────────────────────────

def explore_bundle(results, occ: str, r: int) -> Tuple[str, pd.DataFrame]:
    """Return (markdown header, table of the skills in the optimal r-bundle)."""
    if not results.summary or occ not in results.summary:
        return "Run the pipeline first.", pd.DataFrame()
    curve = reach_curve(results, occ)
    match = curve[curve["r"] == int(r)]
    if match.empty:
        return f"Bundle size r={r} was not evaluated for {occ_display(occ)}.", pd.DataFrame()
    reach_pct = match["reach_pct"].iloc[0]
    skills_str = match["skills"].iloc[0]
    skills = [s.strip() for s in skills_str.split(",") if s.strip()]
    df = pd.DataFrame({"#": range(1, len(skills) + 1), "Skill": skills})
    header = (f"### {occ_display(occ)} — optimal bundle of {r} skills\n"
              f"Reaches **{reach_pct:.0f}%** of job postings.")
    return header, df


# ── 4. Occupation comparison ──────────────────────────────────────────────────

def compare_occupations(results, occ_a: str, occ_b: str, r: int) -> Tuple[str, pd.DataFrame]:
    """Compare the optimal r-bundles of two occupations: shared vs unique skills."""
    if occ_a not in results.summary or occ_b not in results.summary:
        return "Run the pipeline first and pick two occupations.", pd.DataFrame()
    _, df_a = explore_bundle(results, occ_a, r)
    _, df_b = explore_bundle(results, occ_b, r)
    set_a = set(df_a["Skill"]) if not df_a.empty else set()
    set_b = set(df_b["Skill"]) if not df_b.empty else set()
    shared = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    rows = []
    for s in shared:
        rows.append({"Skill": s, "In": "Both"})
    for s in only_a:
        rows.append({"Skill": s, "In": occ_a})
    for s in only_b:
        rows.append({"Skill": s, "In": occ_b})
    table = pd.DataFrame(rows)

    n_shared = len(shared)
    total = len(set_a | set_b)
    overlap_pct = round(n_shared / total * 100) if total else 0
    header = (f"### {occ_display(occ_a)} vs {occ_display(occ_b)} (r={r})\n"
              f"**{n_shared}** shared skills, "
              f"**{len(only_a)}** unique to {occ_a}, "
              f"**{len(only_b)}** unique to {occ_b} "
              f"({overlap_pct}% overlap).")
    return header, table


# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS: Dict[str, dict] = {
    "quick": dict(top_m=12, r_max=8, runs_per_r=2, generations=25, max_pop=3000),
    "balanced": dict(top_m=16, r_max=12, runs_per_r=3, generations=40, max_pop=6000),
    "full": dict(top_m=20, r_max=0, runs_per_r=5, generations=40, max_pop=8000),
}
