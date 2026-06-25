"""
Gradio web UI for the G-TURF pipeline.

Launch with::

    python app.py

then open the printed local URL in a browser. The interface has three tabs:

  1. **Pipeline**    — upload data (or use the synthetic demo), tune every
                       parameter, run HCV -> TURF/GA, and view the figures,
                       summary tables, and a downloadable ZIP of all outputs.
  2. **Statistics**  — CI / Wilcoxon / TOST equivalence analysis on the last run.
  3. **Sensitivity** — sweep a GA hyperparameter and see how robust reach is.

The app is a thin wrapper around the ``gturf`` package: every action maps onto
``GTurfConfig`` + ``run_pipeline`` / ``compute_statistics`` / ``run_sensitivity``,
so the UI and the command-line tools produce identical results.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import traceback
from typing import List, Optional, Tuple

import pandas as pd
import gradio as gr

from gturf import (
    GTurfConfig, run_pipeline, compute_statistics, summarise_statistics,
    run_sensitivity, summarise_sensitivity,
)
from gturf.figures import generate_all_figures, fig_sensitivity
from gturf.sensitivity import DEFAULT_GRIDS
from gturf import io_utils


# ── Shared state ──────────────────────────────────────────────────────────────
# The last pipeline run is cached in-process so the Statistics / Sensitivity tabs
# can reuse the candidate sets without recomputing the whole pipeline.
_STATE: dict = {"results": None, "config": None, "output_dir": None}

PILLARS = ["knowledge", "skills", "traversal"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _new_output_dir() -> str:
    d = tempfile.mkdtemp(prefix="gturf_ui_")
    return d


def _zip_outputs(output_dir: str) -> Optional[str]:
    """Zip the whole output dir; return the zip path (or None)."""
    if not output_dir or not os.path.isdir(output_dir):
        return None
    base = output_dir.rstrip("/")
    archive = shutil.make_archive(base, "zip", output_dir)
    return archive


def _collect_figures(output_dir: str) -> List[str]:
    figs_dir = os.path.join(output_dir, "figures")
    if not os.path.isdir(figs_dir):
        return []
    return [os.path.join(figs_dir, f) for f in sorted(os.listdir(figs_dir))
            if f.lower().endswith(".png")]


def _summary_table(results) -> pd.DataFrame:
    """Concatenate per-occupation reach-vs-r summaries into one tidy table."""
    frames = []
    for occ, df in results.summary.items():
        d = df.copy()
        d.insert(0, "occupation", occ)
        keep = ["occupation", "r", "method", "best_so_far_reach_mean",
                "best_so_far_reach_std", "best_skills"]
        frames.append(d[[c for c in keep if c in d.columns]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Tab 1: Pipeline ───────────────────────────────────────────────────────────

def run_pipeline_ui(
    use_synthetic, oja_file, esco_file, pillar, top_m, hcv_level,
    crossover_rate, mutation_rate, elitism, generations, early_stop_patience,
    min_delta, init_frac, max_pop, runs_per_r, base_seed, exhaustive_threshold,
    r_min, r_max, enforce_monotonicity, occupations_text,
    progress=gr.Progress(track_tqdm=False),
):
    """Run the full pipeline and return (status, summary df, gallery, zip path)."""
    try:
        output_dir = _new_output_dir()

        progress(0.05, desc="Preparing data...")
        if use_synthetic:
            oja_path, esco_path = io_utils.generate_synthetic_dataset(
                os.path.join(output_dir, "synthetic_data")
            )
        else:
            if oja_file is None or esco_file is None:
                return ("⚠️ Please upload BOTH the OJA file and the ESCO mapping file, "
                        "or tick 'Use synthetic demo data'.", None, None, None)
            oja_path, esco_path = oja_file.name, esco_file.name

        # occupations: parse "CODE=URI" lines, else use defaults
        occ = None
        if occupations_text and occupations_text.strip():
            occ = {}
            for line in occupations_text.strip().splitlines():
                if "=" in line:
                    code, uri = line.split("=", 1)
                    occ[code.strip()] = uri.strip()
                elif line.strip():
                    code = line.strip()
                    occ[code] = f"http://data.europa.eu/esco/isco/{code}"

        cfg_kwargs = dict(
            oja_path=oja_path, esco_mapping_path=esco_path, output_dir=output_dir,
            pillar=pillar, top_m=int(top_m), hcv_level=int(hcv_level),
            crossover_rate=float(crossover_rate), mutation_rate=float(mutation_rate),
            elitism=int(elitism), generations=int(generations),
            early_stop_patience=int(early_stop_patience), min_delta=int(min_delta),
            init_frac=float(init_frac), max_pop=int(max_pop),
            runs_per_r=int(runs_per_r), base_seed=int(base_seed),
            exhaustive_threshold=int(exhaustive_threshold), r_min=int(r_min),
            r_max=int(r_max) if r_max and int(r_max) > 0 else None,
            enforce_monotonicity=bool(enforce_monotonicity),
            save_figures=True, save_excel=True,
        )
        if occ:
            cfg_kwargs["occupations"] = occ
        cfg = GTurfConfig(**cfg_kwargs)

        progress(0.15, desc="Running HCV → TURF/GA (this can take a while)...")
        results = run_pipeline(cfg, verbose=False)

        if not results.summary:
            return ("⚠️ No results produced. Check that the occupation codes exist "
                    "in your OJA file and that the candidate set is large enough.",
                    None, None, None)

        progress(0.8, desc="Computing statistics + figures...")
        stats_df = compute_statistics(results)
        if not stats_df.empty:
            stats_df.to_excel(os.path.join(output_dir, "statistics.xlsx"), index=False)
        generate_all_figures(results, cfg, stats_df)

        # cache for the other tabs
        _STATE["results"] = results
        _STATE["config"] = cfg
        _STATE["output_dir"] = output_dir

        progress(0.95, desc="Packaging outputs...")
        zip_path = _zip_outputs(output_dir)
        figs = _collect_figures(output_dir)
        summary = _summary_table(results)

        n_occ = len(results.summary)
        status = (f"✅ Done. Analysed {n_occ} occupation(s), pillar='{cfg.pillar}', "
                  f"M={cfg.top_m}, r={cfg.r_min}..{cfg.r_max}, {cfg.runs_per_r} runs/r. "
                  f"Outputs zipped below; open the Statistics and Sensitivity tabs to "
                  f"analyse this run further.")
        return status, summary, figs, zip_path

    except Exception as exc:  # surface errors to the UI instead of crashing
        return (f"❌ Error: {exc}\n\n{traceback.format_exc()}", None, None, None)


# ── Tab 2: Statistics ─────────────────────────────────────────────────────────

def run_statistics_ui(tost_margin, confidence):
    if _STATE["results"] is None:
        return "⚠️ Run the pipeline first (Pipeline tab).", None, None
    try:
        stats_df = compute_statistics(
            _STATE["results"], tost_margin_pp=float(tost_margin),
            confidence=float(confidence),
        )
        summary = summarise_statistics(stats_df)
        summary_md = "### Headline statistics\n" + "\n".join(
            f"- **{k.replace('_', ' ')}**: {v}" for k, v in summary.items()
        ) if summary else "No statistics available."
        out = os.path.join(_STATE["output_dir"], "statistics_ui.xlsx")
        stats_df.to_excel(out, index=False)
        return summary_md, stats_df, out
    except Exception as exc:
        return f"❌ Error: {exc}", None, None


# ── Tab 3: Sensitivity ────────────────────────────────────────────────────────

def run_sensitivity_ui(parameter, r_value, runs_per_setting):
    if _STATE["results"] is None:
        return "⚠️ Run the pipeline first (Pipeline tab).", None, None, None
    try:
        cfg = _STATE["config"]
        sens_df = run_sensitivity(
            _STATE["results"], cfg, r=int(r_value), parameters=[parameter],
            runs_per_setting=int(runs_per_setting), verbose=False,
        )
        if sens_df.empty:
            return ("⚠️ No sensitivity results (candidate set may be smaller than r).",
                    None, None, None)
        summary = summarise_sensitivity(sens_df)
        cfg.output_dir = _STATE["output_dir"]
        fig_path = fig_sensitivity(sens_df, cfg)
        out = os.path.join(_STATE["output_dir"], f"sensitivity_{parameter}.xlsx")
        sens_df.to_excel(out, index=False)
        status = (f"✅ Swept '{parameter}' at r={r_value}. "
                  f"Smaller reach range ⇒ more robust to this parameter.")
        return status, summary, [fig_path] if fig_path else None, out
    except Exception as exc:
        return f"❌ Error: {exc}", None, None, None


# ── Build the interface ───────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="G-TURF") as demo:
        gr.Markdown(
            "# G-TURF\n"
            "**Optimising skill subset selection via TURF analysis and genetic "
            "algorithms.** Upload an OJA corpus + ESCO mapping (or use the synthetic "
            "demo), tune the parameters, and run the pipeline. All figures and Excel "
            "outputs are downloadable as a ZIP."
        )

        # ── Pipeline tab ──────────────────────────────────────────────────────
        with gr.Tab("Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Data")
                    use_synthetic = gr.Checkbox(
                        label="Use synthetic demo data (no upload needed)", value=True
                    )
                    oja_file = gr.File(label="OJA file (.xlsx / .parquet)",
                                       file_types=[".xlsx", ".parquet"])
                    esco_file = gr.File(label="ESCO mapping file (.xlsx)",
                                        file_types=[".xlsx"])
                    pillar = gr.Dropdown(PILLARS, value="knowledge", label="ESCO pillar")
                    occupations_text = gr.Textbox(
                        label="Occupations (optional, one 'CODE=URI' or 'CODE' per line)",
                        placeholder="C2511\nC2512\nC2513\nC2514",
                        lines=3,
                    )

                    gr.Markdown("### Candidate set")
                    top_m = gr.Slider(5, 40, value=20, step=1, label="Top-M candidate skills")
                    hcv_level = gr.Slider(1, 4, value=4, step=1, label="HCV level")

                    gr.Markdown("### GA hyperparameters")
                    crossover_rate = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Crossover rate (p_c)")
                    mutation_rate = gr.Slider(0.01, 0.5, value=0.25, step=0.01, label="Mutation rate (p_m)")
                    elitism = gr.Slider(0, 5, value=2, step=1, label="Elitism")
                    generations = gr.Slider(10, 100, value=40, step=5, label="Max generations")
                    early_stop_patience = gr.Slider(2, 20, value=8, step=1, label="Early-stop patience")
                    min_delta = gr.Slider(1, 10, value=1, step=1, label="Min improvement (OJAs)")
                    init_frac = gr.Slider(0.05, 1.0, value=0.333, step=0.05, label="Initial population fraction")
                    max_pop = gr.Slider(500, 16000, value=8000, step=500, label="Max population")

                    gr.Markdown("### Experiment control")
                    runs_per_r = gr.Slider(1, 10, value=5, step=1, label="Runs per r")
                    base_seed = gr.Number(value=100, label="Base seed", precision=0)
                    exhaustive_threshold = gr.Slider(2, 6, value=4, step=1, label="Exhaustive threshold (r ≤ this)")
                    r_min = gr.Slider(2, 10, value=2, step=1, label="r min")
                    r_max = gr.Number(value=0, label="r max (0 = M−1)", precision=0)
                    enforce_monotonicity = gr.Checkbox(value=True, label="Enforce monotonic reach curve")

                    run_btn = gr.Button("▶ Run pipeline", variant="primary")

                with gr.Column(scale=2):
                    status = gr.Markdown("Ready. Configure on the left and click **Run pipeline**.")
                    gallery = gr.Gallery(label="Figures", columns=2, height=520)
                    summary_table = gr.Dataframe(label="Reach summary (per occupation, per r)",
                                                 wrap=True)
                    zip_out = gr.File(label="Download all outputs (.zip)")

            run_btn.click(
                run_pipeline_ui,
                inputs=[use_synthetic, oja_file, esco_file, pillar, top_m, hcv_level,
                        crossover_rate, mutation_rate, elitism, generations,
                        early_stop_patience, min_delta, init_frac, max_pop, runs_per_r,
                        base_seed, exhaustive_threshold, r_min, r_max,
                        enforce_monotonicity, occupations_text],
                outputs=[status, summary_table, gallery, zip_out],
            )

        # ── Statistics tab ────────────────────────────────────────────────────
        with gr.Tab("Statistics"):
            gr.Markdown(
                "### Statistical analysis of the last run\n"
                "Computes 95% confidence intervals, a one-sample Wilcoxon signed-rank "
                "test of the GA runs against the deterministic greedy reach, and a "
                "TOST equivalence test. Run the **Pipeline** tab first."
            )
            with gr.Row():
                tost_margin = gr.Slider(0.1, 2.0, value=0.5, step=0.1,
                                        label="TOST equivalence margin (pp)")
                confidence = gr.Slider(0.80, 0.99, value=0.95, step=0.01,
                                       label="Confidence level")
            stats_btn = gr.Button("Compute statistics", variant="primary")
            stats_summary = gr.Markdown()
            stats_table = gr.Dataframe(label="Per-(occupation, r) statistics", wrap=True)
            stats_file = gr.File(label="Download statistics (.xlsx)")
            stats_btn.click(run_statistics_ui, inputs=[tost_margin, confidence],
                            outputs=[stats_summary, stats_table, stats_file])

        # ── Sensitivity tab ───────────────────────────────────────────────────
        with gr.Tab("Sensitivity"):
            gr.Markdown(
                "### GA hyperparameter sensitivity\n"
                "Sweeps a single GA hyperparameter at a fixed bundle size and reports "
                "how much the achieved reach moves. A small range across values means "
                "the result is robust to that parameter. Run the **Pipeline** tab first."
            )
            with gr.Row():
                sens_param = gr.Dropdown(list(DEFAULT_GRIDS.keys()),
                                         value="crossover_rate", label="Parameter to sweep")
                sens_r = gr.Slider(2, 19, value=10, step=1, label="Bundle size r")
                sens_runs = gr.Slider(1, 5, value=3, step=1, label="Runs per setting")
            sens_btn = gr.Button("Run sensitivity sweep", variant="primary")
            sens_status = gr.Markdown()
            sens_summary = gr.Dataframe(label="Reach range per parameter value", wrap=True)
            sens_gallery = gr.Gallery(label="Sensitivity figure", columns=1, height=400)
            sens_file = gr.File(label="Download sensitivity results (.xlsx)")
            sens_btn.click(run_sensitivity_ui, inputs=[sens_param, sens_r, sens_runs],
                           outputs=[sens_status, sens_summary, sens_gallery, sens_file])

        gr.Markdown(
            "---\nG-TURF • SKILLAB (Horizon Europe, GA No. 101132663) • "
            "results are identical to the command-line tools."
        )
    return demo


def main():
    demo = build_app()
    # share=False keeps it local; server_name='0.0.0.0' makes it reachable in a
    # container or on the local network. Theme is passed here for Gradio 6+;
    # older versions accept it on Blocks instead.
    launch_kwargs = dict(server_name="0.0.0.0", server_port=7860)
    try:
        demo.launch(theme=gr.themes.Soft(), **launch_kwargs)
    except TypeError:
        demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
