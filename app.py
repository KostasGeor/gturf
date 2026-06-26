"""
G-TURF web UI — branded, interactive interface for the G-TURF pipeline.

Launch with::

    python app.py

then open the printed local URL. Tabs:

  1. **Run**        — choose a preset or tune every parameter, run the pipeline,
                      and read a plain-language interpretation of the results.
  2. **Explore**    — decision tool ("I can teach N skills -> what coverage?" and
                      the reverse), an interactive bundle explorer, and a
                      side-by-side occupation comparison.
  3. **Statistics** — CI / Wilcoxon / TOST equivalence analysis.
  4. **Sensitivity**— sweep a GA hyperparameter to gauge robustness.

The UI is a thin wrapper over the ``gturf`` package: every action maps onto
``GTurfConfig`` + ``run_pipeline`` / ``compute_statistics`` / ``run_sensitivity``,
so the interface and the command-line tools produce identical results.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import traceback
from typing import List, Optional

import pandas as pd
import gradio as gr

from gturf import (
    GTurfConfig, compute_statistics, summarise_statistics,
    run_sensitivity, summarise_sensitivity,
)
from gturf.figures import generate_all_figures, fig_sensitivity
from gturf.sensitivity import DEFAULT_GRIDS
from gturf import io_utils

import gturf_ui_helpers as H


_STATE: dict = {"results": None, "config": None, "output_dir": None}
PILLARS = ["knowledge", "skills", "traversal"]


# ── Branding ──────────────────────────────────────────────────────────────────

BRAND_PRIMARY = "#2E6FBF"
BRAND_ACCENT = "#E05C2A"

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill=BRAND_PRIMARY,
    button_primary_background_fill_hover="#255a9c",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_label_text_weight="500",
)

CUSTOM_CSS = """
.gturf-hero {
    background: linear-gradient(135deg, #2E6FBF 0%, #1d4e8a 60%, #E05C2A 160%);
    color: white; padding: 26px 30px; border-radius: 14px; margin-bottom: 8px;
}
.gturf-hero h1 { margin: 0 0 6px 0; font-size: 1.9em; font-weight: 700; letter-spacing: -0.5px; }
.gturf-hero p  { margin: 0; opacity: 0.92; font-size: 1.02em; line-height: 1.5; }
.gturf-pill {
    display: inline-block; background: rgba(255,255,255,0.18);
    padding: 3px 11px; border-radius: 20px; font-size: 0.8em; margin-top: 10px;
}
.gturf-card {
    border: 1px solid #e3e8ef; border-radius: 12px; padding: 4px 16px 12px 16px;
    background: #fbfcfe;
}
.gturf-footer { text-align: center; color: #7a8699; font-size: 0.85em; padding-top: 8px; }
"""


# ── Shared helpers ────────────────────────────────────────────────────────────

def _new_output_dir() -> str:
    return tempfile.mkdtemp(prefix="gturf_ui_")


def _zip_outputs(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    return shutil.make_archive(output_dir.rstrip("/"), "zip", output_dir)


def _collect_figures(output_dir: str) -> List[str]:
    figs_dir = os.path.join(output_dir, "figures")
    if not os.path.isdir(figs_dir):
        return []
    return [os.path.join(figs_dir, f) for f in sorted(os.listdir(figs_dir))
            if f.lower().endswith(".png")]


def _run_pipeline_with_progress(cfg: GTurfConfig, progress, base=0.15, span=0.6):
    """Drive the pipeline one occupation at a time so the progress bar advances."""
    from gturf.pipeline import PipelineResults, run_pipeline as _full
    import copy
    occ_codes = list(cfg.occupations.keys())
    n = max(1, len(occ_codes))
    merged: Optional[PipelineResults] = None
    for i, occ in enumerate(occ_codes):
        progress(base + span * (i / n), desc=f"Running {H.occ_display(occ)} ({i+1}/{n})...")
        sub = copy.deepcopy(cfg)
        sub.occupations = {occ: cfg.occupations[occ]}
        res = _full(sub, verbose=False)
        if merged is None:
            merged = res
        else:
            for attr in ("hcv", "candidate_skills", "turf_df", "summary",
                         "run_histories", "greedy", "hcv_vs_turf"):
                getattr(merged, attr).update(getattr(res, attr))
            merged.label_map.update(res.label_map)
    progress(base + span, desc="All occupations done.")
    return merged if merged is not None else PipelineResults(config=cfg)


def _occupation_choices() -> List[str]:
    r = _STATE["results"]
    return list(r.summary.keys()) if r else []


def _r_choices() -> List[int]:
    r = _STATE["results"]
    if not r:
        return []
    occ = next(iter(r.summary))
    return r.summary[occ]["r"].astype(int).tolist()


# ── Presets ───────────────────────────────────────────────────────────────────

def apply_preset(name: str):
    p = H.PRESETS[name]
    return (
        gr.update(value=p["top_m"]),
        gr.update(value=p["r_max"]),
        gr.update(value=p["runs_per_r"]),
        gr.update(value=p["generations"]),
        gr.update(value=p["max_pop"]),
        gr.update(value=f"✨ Applied **{name}** preset. Adjust anything, then click "
                        f"**Run pipeline**."),
    )


# ── Tab 1: Run ────────────────────────────────────────────────────────────────

def run_pipeline_ui(
    use_synthetic, oja_file, esco_file, pillar, top_m, hcv_level,
    crossover_rate, mutation_rate, elitism, generations, early_stop_patience,
    min_delta, init_frac, max_pop, runs_per_r, base_seed, exhaustive_threshold,
    r_min, r_max, enforce_monotonicity, occupations_text,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        output_dir = _new_output_dir()
        progress(0.05, desc="Preparing data...")
        if use_synthetic:
            oja_path, esco_path = io_utils.generate_synthetic_dataset(
                os.path.join(output_dir, "synthetic_data"))
        else:
            if oja_file is None or esco_file is None:
                return ("⚠️ Please upload BOTH files, or tick *Use synthetic demo data*.",
                        None, None, None, "")
            oja_path, esco_path = oja_file.name, esco_file.name

        occ = None
        if occupations_text and occupations_text.strip():
            occ = {}
            for line in occupations_text.strip().splitlines():
                if "=" in line:
                    c, u = line.split("=", 1); occ[c.strip()] = u.strip()
                elif line.strip():
                    c = line.strip(); occ[c] = f"http://data.europa.eu/esco/isco/{c}"

        kwargs = dict(
            oja_path=oja_path, esco_mapping_path=esco_path, output_dir=output_dir,
            pillar=pillar, top_m=int(top_m), hcv_level=int(hcv_level),
            crossover_rate=float(crossover_rate), mutation_rate=float(mutation_rate),
            elitism=int(elitism), generations=int(generations),
            early_stop_patience=int(early_stop_patience), min_delta=int(min_delta),
            init_frac=float(init_frac), max_pop=int(max_pop), runs_per_r=int(runs_per_r),
            base_seed=int(base_seed), exhaustive_threshold=int(exhaustive_threshold),
            r_min=int(r_min), r_max=int(r_max) if r_max and int(r_max) > 0 else None,
            enforce_monotonicity=bool(enforce_monotonicity),
            save_figures=True, save_excel=True,
        )
        if occ:
            kwargs["occupations"] = occ
        cfg = GTurfConfig(**kwargs)

        progress(0.15, desc="Running HCV -> TURF/GA...")
        results = _run_pipeline_with_progress(cfg, progress, base=0.15, span=0.6)
        if not results.summary:
            return ("⚠️ No results. Check occupation codes exist in your data and "
                    "the candidate set is large enough.", None, None, None, "")

        progress(0.8, desc="Statistics + figures...")
        stats_df = compute_statistics(results)
        if not stats_df.empty:
            stats_df.to_excel(os.path.join(output_dir, "statistics.xlsx"), index=False)
        generate_all_figures(results, cfg, stats_df)

        _STATE.update(results=results, config=cfg, output_dir=output_dir)

        progress(0.95, desc="Packaging...")
        interpretation = H.interpret_results(results)
        figs = _collect_figures(output_dir)
        zip_path = _zip_outputs(output_dir)
        status = (f"✅ Done — {len(results.summary)} occupation(s), pillar "
                  f"'{cfg.pillar}', M={cfg.top_m}, r={cfg.r_min}..{cfg.r_max}, "
                  f"{cfg.runs_per_r} runs/r. Head to the **Explore** tab to dig in.")
        return status, interpretation, figs, zip_path, "loaded"
    except Exception as exc:
        return (f"❌ Error: {exc}\n\n```\n{traceback.format_exc()}\n```",
                None, None, None, "")


# ── Tab 2: Explore (decision tool, bundle explorer, comparison) ───────────────

def decision_ui(mode_label, occ, value):
    if _STATE["results"] is None:
        return "⚠️ Run the pipeline first (Run tab)."
    mode = "budget" if mode_label.startswith("I can teach") else "target"
    return H.decision_tool(_STATE["results"], mode, occ, float(value))


def explore_ui(occ, r):
    if _STATE["results"] is None:
        return "⚠️ Run the pipeline first (Run tab).", pd.DataFrame()
    return H.explore_bundle(_STATE["results"], occ, int(r))


def compare_ui(occ_a, occ_b, r):
    if _STATE["results"] is None:
        return "⚠️ Run the pipeline first (Run tab).", pd.DataFrame()
    return H.compare_occupations(_STATE["results"], occ_a, occ_b, int(r))


def refresh_explore_controls():
    """Repopulate dropdowns after a run."""
    occs = _occupation_choices()
    rs = _r_choices()
    occ0 = occs[0] if occs else None
    occ1 = occs[1] if len(occs) > 1 else occ0
    r_mid = rs[len(rs) // 2] if rs else 5
    return (
        gr.update(choices=occs, value=occ0),
        gr.update(choices=occs, value=occ0),
        gr.update(choices=rs, value=r_mid),
        gr.update(choices=occs, value=occ0),
        gr.update(choices=occs, value=occ1),
        gr.update(choices=rs, value=r_mid),
    )


# ── Tab 3: Statistics ─────────────────────────────────────────────────────────

def stats_ui(tost_margin, confidence):
    if _STATE["results"] is None:
        return "⚠️ Run the pipeline first (Run tab).", None, None
    try:
        df = compute_statistics(_STATE["results"], tost_margin_pp=float(tost_margin),
                                confidence=float(confidence))
        summary = summarise_statistics(df)
        md = "### Headline statistics\n" + "\n".join(
            f"- **{k.replace('_', ' ')}**: {v}" for k, v in summary.items()
        ) if summary else "No statistics available."
        out = os.path.join(_STATE["output_dir"], "statistics_ui.xlsx")
        df.to_excel(out, index=False)
        return md, df, out
    except Exception as exc:
        return f"❌ Error: {exc}", None, None


# ── Tab 4: Sensitivity ────────────────────────────────────────────────────────

def sensitivity_ui(parameter, r_value, runs_per_setting):
    if _STATE["results"] is None:
        return "⚠️ Run the pipeline first (Run tab).", None, None, None
    try:
        cfg = _STATE["config"]
        df = run_sensitivity(_STATE["results"], cfg, r=int(r_value),
                             parameters=[parameter],
                             runs_per_setting=int(runs_per_setting), verbose=False)
        if df.empty:
            return "⚠️ No results (candidate set may be smaller than r).", None, None, None
        summary = summarise_sensitivity(df)
        cfg.output_dir = _STATE["output_dir"]
        fig_path = fig_sensitivity(df, cfg)
        out = os.path.join(_STATE["output_dir"], f"sensitivity_{parameter}.xlsx")
        df.to_excel(out, index=False)
        status = (f"✅ Swept **{parameter}** at r={r_value}. "
                  f"A small reach range means the result is robust to this parameter.")
        return status, summary, [fig_path] if fig_path else None, out
    except Exception as exc:
        return f"❌ Error: {exc}", None, None, None


# ── Build the interface ───────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="G-TURF", theme=THEME, css=CUSTOM_CSS) as demo:
        gr.HTML(
            '<div class="gturf-hero">'
            '<h1>G-TURF</h1>'
            '<p>Optimising skill subset selection via TURF analysis and genetic '
            'algorithms. Turn a corpus of job advertisements into compact, '
            'occupation-specific skill bundles that maximise labour-market coverage.</p>'
            '<span class="gturf-pill">SKILLAB · Horizon Europe · GA No. 101132663</span>'
            '</div>'
        )

        # ════ RUN TAB ═════════════════════════════════════════════════════════
        with gr.Tab("🚀 Run"):
            gr.Markdown("#### Start here: pick a preset, or open *Advanced settings* to tune everything.")
            with gr.Row():
                quick_btn = gr.Button("⚡ Quick demo (~30s)", size="sm")
                balanced_btn = gr.Button("⚖️ Balanced", size="sm")
                full_btn = gr.Button("🎯 Full (paper)", size="sm")
            preset_msg = gr.Markdown("")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="gturf-card"):
                        gr.Markdown("### 📁 Data")
                        use_synthetic = gr.Checkbox(
                            label="Use synthetic demo data (no upload needed)", value=True)
                        oja_file = gr.File(label="OJA file (.xlsx / .parquet)",
                                           file_types=[".xlsx", ".parquet"])
                        esco_file = gr.File(label="ESCO mapping (.xlsx)", file_types=[".xlsx"])
                        pillar = gr.Dropdown(PILLARS, value="knowledge", label="ESCO pillar")
                        occupations_text = gr.Textbox(
                            label="Occupations (optional — one 'CODE' or 'CODE=URI' per line)",
                            placeholder="C2511\nC2512\nC2513\nC2514", lines=2)

                    with gr.Accordion("⚙️ Advanced settings", open=False):
                        gr.Markdown("**Candidate set**")
                        top_m = gr.Slider(5, 40, value=12, step=1, label="Top-M candidate skills",
                                          info="How many top-priority skills feed the GA.")
                        hcv_level = gr.Slider(1, 4, value=4, step=1, label="HCV level",
                                              info="ESCO depth used for prioritisation.")
                        gr.Markdown("**GA hyperparameters**")
                        crossover_rate = gr.Slider(0.1, 1.0, value=0.8, step=0.05,
                                                   label="Crossover rate (p_c)",
                                                   info="Chance two parents mix per pairing.")
                        mutation_rate = gr.Slider(0.01, 0.5, value=0.25, step=0.01,
                                                  label="Mutation rate (p_m)",
                                                  info="Chance a bundle swaps one skill.")
                        elitism = gr.Slider(0, 5, value=2, step=1, label="Elitism",
                                            info="Best bundles carried over untouched.")
                        generations = gr.Slider(10, 100, value=25, step=5, label="Max generations",
                                                info="Hard cap; early-stopping usually ends sooner.")
                        early_stop_patience = gr.Slider(2, 20, value=8, step=1,
                                                        label="Early-stop patience")
                        min_delta = gr.Slider(1, 10, value=1, step=1, label="Min improvement (OJAs)")
                        init_frac = gr.Slider(0.05, 1.0, value=0.333, step=0.05,
                                              label="Initial population fraction")
                        max_pop = gr.Slider(500, 16000, value=3000, step=500, label="Max population")
                        gr.Markdown("**Experiment control**")
                        runs_per_r = gr.Slider(1, 10, value=2, step=1, label="Runs per r",
                                               info="More runs = better variance estimates, slower.")
                        base_seed = gr.Number(value=100, label="Base seed", precision=0)
                        exhaustive_threshold = gr.Slider(2, 6, value=4, step=1,
                                                         label="Exhaustive threshold (r ≤ this)")
                        r_min = gr.Slider(2, 10, value=2, step=1, label="r min")
                        r_max = gr.Number(value=8, label="r max (0 = M−1)", precision=0)
                        enforce_monotonicity = gr.Checkbox(value=True,
                                                           label="Enforce monotonic reach curve")

                    run_btn = gr.Button("▶  Run pipeline", variant="primary", size="lg")

                with gr.Column(scale=2):
                    status = gr.Markdown("👋 *Pick a preset above and click* **Run pipeline** "
                                         "*— or use the synthetic demo to try it instantly.*")
                    with gr.Group(elem_classes="gturf-card"):
                        interpretation = gr.Markdown("")
                    gallery = gr.Gallery(label="Figures", columns=2, height=480)
                    zip_out = gr.File(label="⬇️ Download all outputs (.zip)")

            refresh_signal = gr.Textbox(visible=False)

            preset_targets = [top_m, r_max, runs_per_r, generations, max_pop, preset_msg]
            quick_btn.click(lambda: apply_preset("quick"), outputs=preset_targets)
            balanced_btn.click(lambda: apply_preset("balanced"), outputs=preset_targets)
            full_btn.click(lambda: apply_preset("full"), outputs=preset_targets)

        # ════ EXPLORE TAB ═════════════════════════════════════════════════════
        with gr.Tab("🔍 Explore"):
            gr.Markdown("### Make a decision\nUse your real-world constraint to read off the answer.")
            with gr.Group(elem_classes="gturf-card"):
                with gr.Row():
                    dec_mode = gr.Radio(
                        ["I can teach N skills → coverage?", "I want X% coverage → how many skills?"],
                        value="I can teach N skills → coverage?", label="Question")
                    dec_occ = gr.Dropdown([], label="Occupation")
                    dec_value = gr.Number(value=5, label="N skills  (or  X %)", precision=0)
                dec_btn = gr.Button("Answer", variant="primary")
                dec_out = gr.Markdown("")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Bundle explorer\nSee the exact skills in an optimal bundle.")
                    with gr.Group(elem_classes="gturf-card"):
                        exp_occ = gr.Dropdown([], label="Occupation")
                        exp_r = gr.Dropdown([], label="Bundle size r")
                        exp_btn = gr.Button("Show bundle", variant="primary")
                        exp_hdr = gr.Markdown("")
                        exp_table = gr.Dataframe(label="Skills in the optimal bundle", wrap=True)
                with gr.Column():
                    gr.Markdown("### Compare occupations\nShared vs occupation-specific skills.")
                    with gr.Group(elem_classes="gturf-card"):
                        cmp_a = gr.Dropdown([], label="Occupation A")
                        cmp_b = gr.Dropdown([], label="Occupation B")
                        cmp_r = gr.Dropdown([], label="Bundle size r")
                        cmp_btn = gr.Button("Compare", variant="primary")
                        cmp_hdr = gr.Markdown("")
                        cmp_table = gr.Dataframe(label="Skill overlap", wrap=True)

            dec_btn.click(decision_ui, [dec_mode, dec_occ, dec_value], dec_out)
            exp_btn.click(explore_ui, [exp_occ, exp_r], [exp_hdr, exp_table])
            cmp_btn.click(compare_ui, [cmp_a, cmp_b, cmp_r], [cmp_hdr, cmp_table])

        # ════ STATISTICS TAB ══════════════════════════════════════════════════
        with gr.Tab("📊 Statistics"):
            gr.Markdown(
                "### Statistical analysis of the last run\n"
                "95% confidence intervals, a Wilcoxon signed-rank test of the GA runs "
                "against the deterministic greedy reach, and a TOST equivalence test.")
            with gr.Row():
                tost_margin = gr.Slider(0.1, 2.0, value=0.5, step=0.1,
                                        label="TOST equivalence margin (pp)")
                confidence = gr.Slider(0.80, 0.99, value=0.95, step=0.01, label="Confidence level")
            stats_btn = gr.Button("Compute statistics", variant="primary")
            stats_summary = gr.Markdown()
            stats_table = gr.Dataframe(label="Per-(occupation, r) statistics", wrap=True)
            stats_file = gr.File(label="⬇️ Download statistics (.xlsx)")
            stats_btn.click(stats_ui, [tost_margin, confidence],
                            [stats_summary, stats_table, stats_file])

        # ════ SENSITIVITY TAB ═════════════════════════════════════════════════
        with gr.Tab("🎛️ Sensitivity"):
            gr.Markdown(
                "### GA hyperparameter sensitivity\n"
                "Sweep one parameter at a fixed bundle size. A small reach range across "
                "values means the result is robust to that parameter.")
            with gr.Row():
                sens_param = gr.Dropdown(list(DEFAULT_GRIDS.keys()),
                                         value="crossover_rate", label="Parameter")
                sens_r = gr.Slider(2, 19, value=6, step=1, label="Bundle size r")
                sens_runs = gr.Slider(1, 5, value=3, step=1, label="Runs per setting")
            sens_btn = gr.Button("Run sweep", variant="primary")
            sens_status = gr.Markdown()
            sens_summary = gr.Dataframe(label="Reach range per parameter value", wrap=True)
            sens_gallery = gr.Gallery(label="Sensitivity figure", columns=1, height=380)
            sens_file = gr.File(label="⬇️ Download results (.xlsx)")
            sens_btn.click(sensitivity_ui, [sens_param, sens_r, sens_runs],
                           [sens_status, sens_summary, sens_gallery, sens_file])

        gr.HTML('<div class="gturf-footer">G-TURF · results identical to the '
                'command-line tools · MIT licensed</div>')

        run_btn.click(
            run_pipeline_ui,
            inputs=[use_synthetic, oja_file, esco_file, pillar, top_m, hcv_level,
                    crossover_rate, mutation_rate, elitism, generations,
                    early_stop_patience, min_delta, init_frac, max_pop, runs_per_r,
                    base_seed, exhaustive_threshold, r_min, r_max,
                    enforce_monotonicity, occupations_text],
            outputs=[status, interpretation, gallery, zip_out, refresh_signal],
        ).then(
            refresh_explore_controls,
            outputs=[dec_occ, exp_occ, exp_r, cmp_a, cmp_b, cmp_r],
        )

    return demo


def main():
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
