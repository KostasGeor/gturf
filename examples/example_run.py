"""
Minimal end-to-end example using the synthetic dataset.

Run from the repository root:

    python examples/example_run.py

It generates a small synthetic ESCO taxonomy + OJA corpus, runs the full
pipeline, computes statistics, and writes figures into ./example_output/.
Replace the synthetic paths with your own files to analyse real data.
"""
from gturf import GTurfConfig, run_pipeline, compute_statistics, summarise_statistics
from gturf.figures import generate_all_figures
from gturf import io_utils

# 1) Generate synthetic data (swap these two paths for your real files).
oja_path, esco_path = io_utils.generate_synthetic_dataset("example_output/synthetic_data")

# 2) Configure the run. Every parameter is adjustable here.
cfg = GTurfConfig(
    oja_path=oja_path,
    esco_mapping_path=esco_path,
    output_dir="example_output",
    pillar="knowledge",
    top_m=15,
    crossover_rate=0.8,
    mutation_rate=0.25,
    generations=40,
    runs_per_r=5,
    r_max=10,
)

# 3) Run pipeline -> statistics -> figures.
results = run_pipeline(cfg)
stats = compute_statistics(results, tost_margin_pp=0.5)
print(summarise_statistics(stats))
generate_all_figures(results, cfg, stats)
print("See example_output/ for Excel files and figures.")
