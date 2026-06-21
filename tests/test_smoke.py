"""
Smoke tests for the G-TURF package using the synthetic dataset.

Run with::

    pytest tests/ -q
"""
import os
import tempfile

import numpy as np

from gturf import GTurfConfig, run_pipeline, compute_statistics
from gturf import io_utils
from gturf.turf import vectorized_reach


def _synthetic_cfg(tmp):
    oja, esco = io_utils.generate_synthetic_dataset(
        os.path.join(tmp, "data"), ojas_per_occupation=300, seed=0
    )
    return GTurfConfig(
        oja_path=oja, esco_mapping_path=esco, output_dir=tmp,
        pillar="knowledge", top_m=10, r_max=6, runs_per_r=3,
        save_figures=False, save_excel=False,
    )


def test_vectorized_reach_counts_ojas():
    # 3 OJAs x 2 skills; bundle {skill0} covers OJAs 0 and 2.
    X = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.bool_)
    pop = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    reach = vectorized_reach(X, pop)
    assert list(reach) == [2, 2, 3]


def test_pipeline_runs_and_is_monotone():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _synthetic_cfg(tmp)
        results = run_pipeline(cfg, verbose=False)
        assert results.summary, "no occupations produced results"
        for occ, summary in results.summary.items():
            reach = summary["best_so_far_reach_mean"].values
            # monotonic non-decreasing reach in r
            assert np.all(np.diff(reach) >= -1e-9), f"{occ} reach not monotone"


def test_exhaustive_matches_or_exceeds_greedy_at_small_r():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _synthetic_cfg(tmp)
        results = run_pipeline(cfg, verbose=False)
        for occ in results.summary:
            summ = results.summary[occ].set_index("r")
            greedy = results.greedy[occ].set_index("r")
            for r in [2, 3, 4]:
                if r in summ.index and r in greedy.index:
                    # exhaustive optimum must be >= greedy
                    assert summ.loc[r, "best_so_far_reach_mean"] + 1e-6 >= greedy.loc[r, "reach"]


def test_statistics_have_expected_columns():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _synthetic_cfg(tmp)
        results = run_pipeline(cfg, verbose=False)
        stats = compute_statistics(results)
        for col in ["occupation", "r", "mean_pct", "wilcoxon_p", "tost_p", "equivalent"]:
            assert col in stats.columns


def test_config_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = GTurfConfig(top_m=15, crossover_rate=0.7, pillar="skills")
        path = os.path.join(tmp, "cfg.json")
        cfg.to_json(path)
        cfg2 = GTurfConfig.from_json(path)
        assert cfg2.top_m == 15
        assert cfg2.crossover_rate == 0.7
        assert cfg2.pillar == "skills"
