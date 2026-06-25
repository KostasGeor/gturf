# G-TURF

[![CI](https://github.com/your-org/gturf/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/gturf/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**Optimising skill subset selection via TURF analysis and genetic algorithms.**

G-TURF is a reproducible Python pipeline that turns a corpus of Online Job
Advertisements (OJAs) into compact, occupation-specific **skill bundles** that
maximise labour-market coverage. It combines two ideas the literature usually
treats separately:

1. **Which skills matter most** for an occupation — answered by **Hierarchical
   Cumulative Voting (HCV)** over the ESCO taxonomy.
2. **Which combination of a fixed number of skills covers the most job
   postings** — answered by **TURF analysis** optimised with a **Genetic
   Algorithm (GA)**, with exhaustive search used as a ground-truth baseline at
   small bundle sizes.

The package reproduces every quantitative artefact in the accompanying paper —
HCV priority rankings, reach-versus-bundle-size curves, the greedy-TURF and
random baselines, the exhaustive validation, the computational-scaling analysis,
the statistical analysis (confidence intervals, Wilcoxon and TOST equivalence
tests), and the hyperparameter sensitivity analysis — for **any** dataset and
**any** parameter configuration.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Input data format](#input-data-format)
- [Command-line usage](#command-line-usage)
- [Adjustable parameters](#adjustable-parameters)
- [Using G-TURF as a library](#using-g-turf-as-a-library)
- [Outputs](#outputs)
- [How the pipeline works](#how-the-pipeline-works)
- [Reproducing the paper](#reproducing-the-paper)
- [Testing](#testing)
- [Citation](#citation)
- [License](#license)

---

## Installation

Requires Python 3.9+.

**Option A — pip install (recommended).** Installs the library and three
command-line tools (`gturf-run`, `gturf-sensitivity`, `gturf-statistics`):

```bash
git clone https://github.com/your-org/gturf.git
cd gturf
pip install -e .
```

**Option B — plain scripts.** Install the dependencies and run the scripts
directly without installing the package:

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --help
```

For Parquet OJA inputs, also install the optional extra: `pip install -e ".[parquet]"`.

---

## Quick start

You can try the entire pipeline **without any data** — a synthetic ESCO
taxonomy and OJA corpus are generated for you:

```bash
gturf-run --synthetic --output-dir demo_run
```

or, without installing:

```bash
python scripts/run_pipeline.py --synthetic --output-dir demo_run
```

This runs HCV → TURF/GA for four synthetic occupations, computes statistics, and
writes Excel files and figures into `demo_run/`. Once it works, point it at your
own data (see below).

---

## Input data format

G-TURF needs two files. Neither is redistributed with this repository — you
supply your own. (The proprietary EURES/ESCO files used in the paper are not
included.)

### 1. ESCO taxonomy mapping (`--esco-mapping`)

One row per ESCO concept. Required columns:

| Column | Description |
|---|---|
| `conceptUri` | Unique ESCO concept URI (string) |
| `preferredLabel` | Human-readable skill name |
| `children` | Stringified Python list of child concept URIs |
| `skills_levels`, `knowledge_levels`, `traversal_levels` | Stringified list of levels at which the concept appears, per pillar |
| `skills_ancestors`, `knowledge_ancestors`, `traversal_ancestors` | Stringified list of ancestor-URI lists, per pillar |

List-valued columns are stored as strings (as Excel does) and parsed with
`ast.literal_eval`.

### 2. OJA corpus (`--oja`)

Either of two schemas is accepted:

- **Generic schema** (Excel or Parquet): columns `occupation` (a code such as
  `C2511`) and `esco_skills` (a stringified list of ESCO skill URIs demanded by
  that posting).
- **Project schema**: if a module named `splitting` exposing
  `load_skills(path)` is importable, it is used automatically (this matches the
  original research layout).

To see a concrete, valid example of both files, run the synthetic generator and
inspect what it writes:

```bash
python -c "from gturf import io_utils; io_utils.generate_synthetic_dataset('sample_data')"
```

---

## Command-line usage

Three commands are installed. Each has `--help`.

### Main pipeline — `gturf-run`

```bash
gturf-run \
    --oja jobs_software_engineer.xlsx \
    --esco-mapping new_ESCO_mapping.xlsx \
    --pillar knowledge \
    --output-dir results_knowledge
```

### Sensitivity analysis — `gturf-sensitivity`

Sweeps each GA hyperparameter one at a time at a chosen bundle size and reports
how much the reach moves (small range ⇒ robust):

```bash
gturf-sensitivity \
    --oja jobs_software_engineer.xlsx --esco-mapping new_ESCO_mapping.xlsx \
    --pillar knowledge --r 10 --runs-per-setting 3 \
    --parameters crossover_rate mutation_rate generations max_pop \
    --output-dir sensitivity_knowledge
```

### Statistical analysis — `gturf-statistics`

Computes mean / std / 95% CI per (occupation, r), a Wilcoxon signed-rank test of
the GA runs against the deterministic greedy reach, and a TOST equivalence test:

```bash
gturf-statistics \
    --oja jobs_software_engineer.xlsx --esco-mapping new_ESCO_mapping.xlsx \
    --pillar knowledge --tost-margin 0.5 --runs-per-r 5 \
    --output-dir statistics_knowledge
```

> **Tip:** add `--synthetic` to any of the three commands to try them with no
> input files.

---

## Adjustable parameters

Every parameter is a command-line flag — you never edit source code. The full
list (with defaults matching the paper):

| Flag | Default | Meaning |
|---|---|---|
| `--pillar` | `knowledge` | ESCO pillar: `skills`, `knowledge`, or `traversal` |
| `--top-m` | `20` | Number of top-priority skills kept as the GA candidate set (M) |
| `--hcv-level` | `4` | ESCO level whose ranked skills feed TURF |
| `--crossover-rate` | `0.8` | Uniform-crossover probability (p_c) |
| `--mutation-rate` | `0.25` | Swap-mutation probability (p_m) |
| `--elitism` | `2` | Individuals carried unchanged each generation |
| `--generations` | `40` | Hard cap on GA generations (G_max) |
| `--early-stop-patience` | `8` | Stop after this many no-improvement generations |
| `--min-delta` | `1` | Minimum reach gain counted as progress |
| `--init-frac` | `0.333` | Fraction of the search space sampled as the initial population |
| `--max-pop` | `8000` | Hard cap on initial population size |
| `--runs-per-r` | `5` | Independent GA runs per (occupation, r) |
| `--base-seed` | `100` | Base RNG seed (per-run seeds derived from it) |
| `--exhaustive-threshold` | `4` | r ≤ this uses exhaustive search; larger r uses the GA |
| `--r-min` / `--r-max` | `2` / `M−1` | Range of bundle sizes evaluated |
| `--no-monotonicity` | off | Disable monotonic reach-curve enforcement |
| `--occupations` | four SE codes | Which occupation codes to analyse |
| `--no-figures`, `--no-excel`, `--no-stats` | off | Skip parts of the output |

---

## Using G-TURF as a library

```python
from gturf import GTurfConfig, run_pipeline, compute_statistics, generate_all_figures

cfg = GTurfConfig(
    oja_path="jobs_software_engineer.xlsx",
    esco_mapping_path="new_ESCO_mapping.xlsx",
    pillar="knowledge",
    top_m=20,
    crossover_rate=0.8,
    mutation_rate=0.25,
    runs_per_r=5,
)

results = run_pipeline(cfg)                       # HCV -> TURF/GA per occupation
stats   = compute_statistics(results, tost_margin_pp=0.5)
generate_all_figures(results, cfg, stats)

# Inspect programmatically:
results.candidate_skills["C2511"]   # top-M skill URIs for an occupation
results.summary["C2511"]            # reach per bundle size r
results.greedy["C2511"]             # greedy-TURF baseline
results.hcv_vs_turf["C2511"]        # HCV top-r naive baseline vs G-TURF
```

The full configuration of every run is written to `run_config.json` in the
output directory, so any result can be reproduced exactly.

---

## Outputs

For each occupation (e.g. `C2511/`):

- `HCV_results_<occ>.xlsx` — full HCV priority table (all levels).
- `EXHAUSTIVE_r{2,3,4}.xlsx` — guaranteed-optimal bundles at small r.
- `GA_r{r}_run{i}_seed{s}.xlsx` — per-run GA history (one file per run).
- `summary_by_r_<occ>.xlsx` — best reach per bundle size, with mean/std and timing.
- `greedy_turf_<occ>.xlsx` — greedy-TURF baseline.
- `hcv_vs_turf_comparison_<occ>.xlsx` — HCV-top-r naive baseline vs G-TURF.

At the top level:

- `statistics.xlsx` — CI / Wilcoxon / TOST per (occupation, r).
- `run_config.json` — exact parameters used (reproducibility record).
- `figures/` — `fig_hcv_heatmap`, `fig_reach_vs_r`, `fig_gturf_vs_greedy`,
  `fig_hcv_vs_gturf`, `fig_bundle_composition`, `fig_computational_scaling`
  (and `fig_sensitivity` from the sensitivity command).

---

## How the pipeline works

```
OJA corpus + ESCO taxonomy
        │
        ▼
┌─────────────────────┐   Hierarchical Cumulative Voting:
│  Stage 1 — HCV      │   propagate skill frequencies through the taxonomy,
│                     │   producing normalised, occupation-specific priorities.
└─────────────────────┘   Output: top-M candidate skills at level L.
        │
        ▼
┌─────────────────────┐   For each bundle size r:
│  Stage 2 — TURF/GA  │     r ≤ threshold → exhaustive search (global optimum)
│                     │     r >  threshold → genetic algorithm (reach fitness)
└─────────────────────┘   Output: maximum-reach bundle per r.
        │
        ▼
  Baselines + analysis: greedy TURF, HCV-top-r, random,
  exhaustive validation, statistics, sensitivity, figures.
```

**Reach** is the *unduplicated* number of OJAs covered by a bundle: a posting
counts once if it contains at least one skill from the bundle. Reach is monotone
and submodular, so the greedy heuristic is provably near-optimal for the
single-objective case; the GA matches it where verifiable and remains tractable
where exhaustive search does not, while also extending to constrained or
multi-objective bundle selection.

---

## Web UI

A [Gradio](https://gradio.app) web interface wraps the whole pipeline so you can
run G-TURF from a browser — no command line needed.

```bash
pip install -r requirements-app.txt   # installs gturf + gradio
python app.py                          # then open http://localhost:7860
```

The interface has three tabs:

- **Pipeline** — upload an OJA file and ESCO mapping (or tick *Use synthetic demo
  data*), tune every parameter with sliders, run HCV → TURF/GA, and view the
  figures and reach summary. All outputs download as a single ZIP.
- **Statistics** — 95% CI, Wilcoxon, and TOST equivalence analysis of the last run.
- **Sensitivity** — sweep a GA hyperparameter and see how robust the reach is.

Results are identical to the command-line tools — the UI is a thin wrapper around
the same `gturf` package.

### Docker / Hugging Face Spaces

A `Dockerfile` is included for containerised deployment:

```bash
docker build -t gturf-app .
docker run -p 7860:7860 gturf-app
```

The image exposes port 7860 and runs as-is on Hugging Face Spaces (Docker SDK).

## Reproducing the paper

With the paper's data files in place:

```bash
# Knowledge pillar (main results)
gturf-run --oja jobs_software_engineer.xlsx --esco-mapping new_ESCO_mapping.xlsx \
          --pillar knowledge --output-dir results_knowledge

# Skills pillar
gturf-run --oja jobs_software_engineer.xlsx --esco-mapping new_ESCO_mapping.xlsx \
          --pillar skills --output-dir results_skills

# Statistics and sensitivity
gturf-statistics  --oja ... --esco-mapping ... --pillar knowledge --output-dir stats_knowledge
gturf-sensitivity --oja ... --esco-mapping ... --pillar knowledge --r 10 --output-dir sens_knowledge
```

Default parameters reproduce the paper. Results are deterministic given the
`--base-seed`.

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

The tests run the full pipeline on synthetic data and check core invariants
(reach monotonicity, exhaustive ≥ greedy at small r, the vectorised reach
computation, and config round-tripping).

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development
setup, conventions, and how to submit changes. In short: `pip install -e ".[dev]"`,
make your change with a test, and ensure `pytest tests/ -q` passes.

## Citation

If you use G-TURF, please cite:

> Georgiou, K., Mittas, N., & Angelis, L. (2026). *G-TURF: Optimising skill
> subset selection via TURF analysis and genetic algorithms.*

The predecessor method:

> Ntaoulas, V., Georgiou, K., Mittas, N., & Angelis, L. (2025). *H-TURF:
> Detecting Optimal Green Software Engineering Skillsets Using TURF Analysis and
> Hierarchical Cumulative Voting.* Euromicro SEAA.

---

## License

MIT — see [LICENSE](LICENSE).

This work was developed in the context of the **SKILLAB** Horizon Europe project
(Grant Agreement No. 101132663).
