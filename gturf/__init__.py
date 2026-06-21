"""
G-TURF: Optimising skill subset selection via TURF analysis and genetic algorithms.

A reproducible pipeline that turns a corpus of Online Job Advertisements (OJAs)
into compact, occupation-specific skill bundles maximising labour-market coverage.

Stages
-------
1. **HCV** (:mod:`gturf.hcv`)         — Hierarchical Cumulative Voting prioritises
   skills over the ESCO taxonomy.
2. **TURF + GA** (:mod:`gturf.turf`,
   :mod:`gturf.ga`)                    — find the r-skill bundle of maximum
   unduplicated reach (exhaustive for small r, genetic algorithm for larger r).

Analysis add-ons
----------------
- :mod:`gturf.statistics`  — CI, Wilcoxon, TOST equivalence tests.
- :mod:`gturf.sensitivity` — hyperparameter sensitivity sweeps.
- :mod:`gturf.figures`     — publication figures.

Typical use
-----------
>>> from gturf import GTurfConfig, run_pipeline
>>> cfg = GTurfConfig(oja_path="ojas.xlsx", esco_mapping_path="esco.xlsx",
...                   pillar="knowledge")
>>> results = run_pipeline(cfg)
"""
from .config import GTurfConfig
from .pipeline import run_pipeline, PipelineResults
from .statistics import compute_statistics, summarise_statistics
from .sensitivity import run_sensitivity, summarise_sensitivity
from .figures import generate_all_figures
from . import io_utils

__version__ = "1.0.0"

__all__ = [
    "GTurfConfig",
    "run_pipeline",
    "PipelineResults",
    "compute_statistics",
    "summarise_statistics",
    "run_sensitivity",
    "summarise_sensitivity",
    "generate_all_figures",
    "io_utils",
]
