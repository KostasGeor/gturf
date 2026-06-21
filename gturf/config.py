"""
Configuration for the G-TURF pipeline.

All tunable parameters live in the :class:`GTurfConfig` dataclass. Nothing in the
pipeline reads a global variable directly; every stage receives a config object,
which makes runs fully reproducible and easy to sweep in the sensitivity analysis.

The command-line scripts in ``scripts/`` build a :class:`GTurfConfig` from
argparse arguments, so end users never need to edit source code to change a
parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


@dataclass
class GTurfConfig:
    """All parameters that control a G-TURF run.

    Attributes are grouped into: data/paths, HCV stage, candidate-set,
    GA hyperparameters, experiment control, and output.
    """

    # ── Data and paths ────────────────────────────────────────────────────────
    oja_path: str = "jobs_software_engineer.xlsx"
    """Path to the Excel file with one row per OJA and the ESCO skill lists."""

    esco_mapping_path: str = "new_ESCO_mapping.xlsx"
    """Path to the ESCO taxonomy mapping file (concept URIs, levels, ancestors,
    children, preferred labels)."""

    output_dir: str = "gturf_output"
    """Directory where all Excel files and figures are written."""

    pillar: str = "knowledge"
    """Which ESCO pillar to analyse: 'skills', 'knowledge', or 'traversal'.
    Must match the column-name prefix used in the ESCO mapping file
    (e.g. 'knowledge' -> 'knowledge_levels', 'knowledge_ancestors')."""

    # Occupation code -> ESCO ISCO URI. Override to analyse any set of occupations.
    occupations: Dict[str, str] = field(default_factory=lambda: {
        "C2511": "http://data.europa.eu/esco/isco/C2511",
        "C2512": "http://data.europa.eu/esco/isco/C2512",
        "C2513": "http://data.europa.eu/esco/isco/C2513",
        "C2514": "http://data.europa.eu/esco/isco/C2514",
    })

    # ── Candidate set (HCV output) ────────────────────────────────────────────
    top_m: int = 20
    """Number of top-priority level-L skills kept as the GA candidate set (M)."""

    hcv_level: int = 4
    """ESCO level whose ranked skills feed the TURF stage (1-indexed)."""

    # ── GA hyperparameters ────────────────────────────────────────────────────
    crossover_rate: float = 0.8
    """Uniform-crossover probability (p_c)."""

    mutation_rate: float = 0.25
    """Swap-mutation probability (p_m)."""

    elitism: int = 2
    """Number of top individuals carried unchanged to the next generation."""

    generations: int = 40
    """Hard upper bound on GA generations (G_max). Early stopping usually
    triggers first."""

    early_stop_patience: int = 8
    """Stop if best reach does not improve for this many consecutive generations."""

    min_delta: int = 1
    """Minimum reach improvement (in OJAs) that counts as progress for early
    stopping."""

    init_frac: float = 1.0 / 3.0
    """Fraction of the full combinatorial space sampled as the initial GA
    population."""

    max_pop: int = 8000
    """Hard cap on the initial population size, keeping large-r runs tractable."""

    # ── Experiment control ────────────────────────────────────────────────────
    runs_per_r: int = 5
    """Independent GA runs per (occupation, r) for variance estimation."""

    base_seed: int = 100
    """Base RNG seed. Per-run seeds are derived deterministically from this."""

    exhaustive_threshold: int = 4
    """Bundle sizes r <= this value are solved by exhaustive search (guaranteed
    optimum); larger r use the GA."""

    r_min: int = 2
    """Smallest bundle size evaluated."""

    r_max: Optional[int] = None
    """Largest bundle size evaluated. If None, defaults to top_m - 1."""

    enforce_monotonicity: bool = True
    """Propagate the best-so-far reach forward so reach-vs-r curves never
    decrease (adding skills can never reduce coverage)."""

    # ── Output control ────────────────────────────────────────────────────────
    save_figures: bool = True
    save_excel: bool = True
    figure_dpi: int = 200

    def __post_init__(self):
        if self.r_max is None:
            self.r_max = self.top_m - 1
        self.pillar = self.pillar.lower().strip()
        if self.pillar not in {"skills", "knowledge", "traversal"}:
            raise ValueError(
                f"pillar must be 'skills', 'knowledge' or 'traversal', got '{self.pillar}'"
            )
        if not (0.0 < self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in (0, 1]")
        if not (0.0 < self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in (0, 1]")
        if self.r_min < 2:
            raise ValueError("r_min must be >= 2")
        if self.r_max >= self.top_m:
            raise ValueError("r_max must be < top_m")

    @property
    def r_range(self) -> List[int]:
        return list(range(self.r_min, self.r_max + 1))

    @property
    def levels_column(self) -> str:
        return f"{self.pillar}_levels"

    @property
    def ancestors_column(self) -> str:
        return f"{self.pillar}_ancestors"

    def to_json(self, path: str) -> None:
        """Persist the exact config used for a run (reproducibility record)."""
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "GTurfConfig":
        with open(path) as fh:
            data = json.load(fh)
        return cls(**data)
