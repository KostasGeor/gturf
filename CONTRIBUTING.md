# Contributing to G-TURF

Thanks for your interest in improving G-TURF! This document explains how to set
up a development environment, the conventions the codebase follows, and how to
propose changes.

## Development setup

```bash
git clone https://github.com/your-org/gturf.git
cd gturf
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -e ".[dev]"
```

Verify everything works before making changes:

```bash
pytest tests/ -q
gturf-run --synthetic --output-dir /tmp/smoke   # full pipeline on synthetic data
```

## How the package is organised

Each stage of the pipeline lives in its own module, and nothing reads global
state — every function receives the `GTurfConfig` object or explicit arguments.
This keeps runs reproducible and makes the sensitivity analysis possible.

| Module | Responsibility |
|---|---|
| `gturf/config.py` | All tunable parameters (`GTurfConfig` dataclass) |
| `gturf/io_utils.py` | Loading inputs, validating schema, synthetic data |
| `gturf/hcv.py` | Hierarchical Cumulative Voting |
| `gturf/turf.py` | Reach, exhaustive search, greedy baseline |
| `gturf/ga.py` | The genetic algorithm and its operators |
| `gturf/pipeline.py` | Orchestration across occupations and bundle sizes |
| `gturf/statistics.py` | CI, Wilcoxon, TOST |
| `gturf/sensitivity.py` | Hyperparameter sweeps |
| `gturf/figures.py` | Figure generation |
| `gturf/cli.py` | Command-line entry points |

## Conventions

- **Add a parameter the right way.** New tunables go on `GTurfConfig` with a
  sensible default and a one-line docstring, then get a matching `--flag` in the
  relevant parser in `gturf/cli.py`. Never hard-code values inside the stages.
- **Determinism.** Anything stochastic must derive its seed from
  `cfg.base_seed`. A run with a fixed seed must reproduce exactly.
- **Keep stages pure.** Stage functions take inputs and return values; file I/O
  belongs in `pipeline.py` / `io_utils.py`, not inside `hcv.py`, `ga.py`, etc.
- **Docstrings.** Public functions get a short docstring explaining what they do
  and what they return. The math should match the paper's notation.
- **Style.** Follow PEP 8; keep lines readable. No enforced formatter is
  required, but consistency with the surrounding code is appreciated.

## Tests

Every behavioural change should keep the existing tests green and, where it adds
new behaviour, include a test. The suite runs on the synthetic dataset so it
needs no proprietary data:

```bash
pytest tests/ -q
```

Useful invariants already covered (good models for new tests): reach is monotone
in `r`; exhaustive search ≥ greedy at small `r`; the vectorised reach matches a
hand-computed example; config round-trips through JSON.

## Submitting changes

1. Fork the repository and create a branch:
   `git checkout -b feature/short-description`.
2. Make your change, add or update tests, and run `pytest tests/ -q`.
3. Run the synthetic pipeline once to confirm nothing broke end to end.
4. Open a pull request describing **what** changed and **why**. If it affects
   results or output files, say so explicitly.

## Reporting issues

When opening an issue, please include: the command you ran (with flags), the
full error message or unexpected output, your Python version, and whether it
reproduces with `--synthetic`. A reproduction on synthetic data is the fastest
path to a fix.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
