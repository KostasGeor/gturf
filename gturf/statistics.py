"""
Statistical analysis of G-TURF results.

Provides:
  - per-(occupation, r) mean / std / 95% CI across the independent GA runs;
  - a one-sample Wilcoxon signed-rank test of GA runs vs the deterministic
    greedy reach (tests whether a difference exists);
  - a TOST equivalence test (tests whether the methods are equivalent within a
    pre-specified margin).

The two tests are complementary: a non-significant Wilcoxon result alone only
means "no difference detected", whereas TOST provides positive evidence of
equivalence within the margin.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from .pipeline import PipelineResults


def per_run_reaches(results: PipelineResults, occ: str, r: int) -> List[float]:
    """Final best reach of each independent run at (occ, r)."""
    histories = results.run_histories.get(occ, {}).get(r, [])
    return [float(df["best_so_far_reach"].iloc[-1]) for df in histories]


def compute_statistics(
    results: PipelineResults,
    tost_margin_pp: float = 0.5,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Return a tidy dataframe with mean, std, CI, Wilcoxon p, and TOST result
    for every (occupation, r). ``tost_margin_pp`` is the equivalence margin in
    percentage points of reach.
    """
    rows = []
    alpha = 1.0 - confidence
    for occ in results.summary:
        n_jobs = len(results.turf_df.get(occ, []))
        greedy = results.greedy.get(occ)
        greedy_by_r = greedy.set_index("r")["reach"].to_dict() if greedy is not None else {}
        for r in results.summary[occ]["r"]:
            reaches = per_run_reaches(results, occ, int(r))
            if len(reaches) < 2:
                continue
            arr = np.array(reaches, dtype=float)
            mean, std, k = arr.mean(), arr.std(ddof=1), len(arr)
            se = std / np.sqrt(k)
            tcv = stats.t.ppf(1 - alpha / 2, df=k - 1)
            ci_lo, ci_hi = mean - tcv * se, mean + tcv * se

            greedy_reach = float(greedy_by_r.get(int(r), np.nan))
            wil_p, tost_p, equivalent = np.nan, np.nan, None
            if not np.isnan(greedy_reach):
                diffs = arr - greedy_reach
                if std == 0:
                    # all runs identical to each other
                    equivalent = abs(mean - greedy_reach) <= tost_margin_pp / 100 * n_jobs
                    tost_p = 0.0 if equivalent else 1.0
                else:
                    try:
                        _, wil_p = stats.wilcoxon(diffs, alternative="two-sided")
                    except ValueError:
                        wil_p = np.nan
                    margin_abs = tost_margin_pp / 100 * n_jobs
                    _, p_lo = stats.ttest_1samp(diffs, -margin_abs, alternative="greater")
                    _, p_hi = stats.ttest_1samp(diffs, margin_abs, alternative="less")
                    tost_p = max(p_lo, p_hi)
                    equivalent = tost_p < alpha

            rows.append({
                "occupation": occ,
                "r": int(r),
                "n_runs": k,
                "mean_reach": round(mean, 3),
                "std_reach": round(std, 4),
                "mean_pct": round(mean / n_jobs * 100, 3) if n_jobs else np.nan,
                "std_pct": round(std / n_jobs * 100, 4) if n_jobs else np.nan,
                "ci95_low_pct": round(ci_lo / n_jobs * 100, 3) if n_jobs else np.nan,
                "ci95_high_pct": round(ci_hi / n_jobs * 100, 3) if n_jobs else np.nan,
                "greedy_reach": greedy_reach,
                "diff_pp": round((mean - greedy_reach) / n_jobs * 100, 3)
                if n_jobs and not np.isnan(greedy_reach) else np.nan,
                "wilcoxon_p": round(wil_p, 4) if not np.isnan(wil_p) else np.nan,
                "tost_p": round(tost_p, 4) if not np.isnan(tost_p) else np.nan,
                "equivalent": equivalent,
            })
    return pd.DataFrame(rows)


def summarise_statistics(stats_df: pd.DataFrame) -> Dict[str, float]:
    """Headline numbers for the paper / logs."""
    if stats_df.empty:
        return {}
    return {
        "configs": int(len(stats_df)),
        "zero_variance_configs": int((stats_df["std_reach"] == 0).sum()),
        "max_std_pp": float(stats_df["std_pct"].max()),
        "max_ci_width_pp": float((stats_df["ci95_high_pct"] - stats_df["ci95_low_pct"]).max()),
        "wilcoxon_significant": int((stats_df["wilcoxon_p"] < 0.05).sum()),
        "tost_equivalent": int((stats_df["equivalent"] == True).sum()),  # noqa: E712
    }
