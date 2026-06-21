"""
Input/output utilities for the G-TURF pipeline.

Handles loading the two required inputs (the OJA corpus and the ESCO taxonomy
mapping), validating their schema, and a synthetic-data generator so users can
exercise the full pipeline without the proprietary EURES files.

Expected input schema
----------------------
**ESCO mapping file** (``esco_mapping_path``): one row per ESCO concept, columns:

  - ``conceptUri``            : unique ESCO URI (str)
  - ``preferredLabel``        : human-readable skill name (str)
  - ``children``              : list of child concept URIs (stringified Python list)
  - ``<pillar>_levels``       : list of levels at which the concept appears
  - ``<pillar>_ancestors``    : list of ancestor-URI lists
    (for each of skills / knowledge / traversal)

**OJA file** (``oja_path``): used by :func:`load_oja_skill_lists`, which delegates
to a user-supplied ``splitting.load_skills`` if present, or reads a simple schema
(see :func:`load_oja_skill_lists`).
"""
from __future__ import annotations

import ast
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ── ESCO mapping ──────────────────────────────────────────────────────────────

LIST_COLUMNS = [
    "skills_levels", "knowledge_levels", "traversal_levels",
    "skills_ancestors", "knowledge_ancestors", "traversal_ancestors",
    "children",
]


def load_esco_mapping(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load the ESCO mapping file and return (dataframe, uri->label dict).

    List-valued columns are stored as strings in Excel; they are parsed with
    ``ast.literal_eval`` (safer than ``eval``).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ESCO mapping file not found: {path}\n"
            "Provide --esco-mapping or generate synthetic data with "
            "`gturf-generate-sample`."
        )
    df = pd.read_excel(path)

    missing = [c for c in ["conceptUri", "preferredLabel", "children"] if c not in df.columns]
    if missing:
        raise ValueError(f"ESCO mapping missing required columns: {missing}")

    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_safe_eval)

    label_map = {row["conceptUri"]: row["preferredLabel"] for _, row in df.iterrows()}
    label_map[None] = ""
    return df, label_map


def _safe_eval(value):
    """Parse a stringified Python list; pass through if already a list."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


# ── OJA corpus ────────────────────────────────────────────────────────────────

def load_oja_skill_lists(path: str, occupations: Dict[str, str]) -> Dict[str, List[List[str]]]:
    """Return {occupation_code: [list of ESCO-URI lists, one per OJA]}.

    Resolution order:

    1. If a module named ``splitting`` exposing ``load_skills(path)`` is importable
       (the original project layout), it is used and its four returned lists are
       mapped onto the first four occupation codes in ``occupations``.
    2. Otherwise the file is read with the generic schema: an Excel/Parquet file
       with at least two columns, ``occupation`` (a code matching the keys of
       ``occupations``) and ``esco_skills`` (a stringified list of skill URIs).
    """
    # Strategy 1 — original project helper
    try:
        from splitting import load_skills  # type: ignore
        lists = load_skills(path)
        codes = list(occupations.keys())
        return {codes[i]: lists[i] for i in range(min(len(codes), len(lists)))}
    except Exception:
        pass

    # Strategy 2 — generic schema
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"OJA file not found: {path}\n"
            "Provide --oja or generate synthetic data with `gturf-generate-sample`."
        )
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path)

    if "occupation" not in df.columns or "esco_skills" not in df.columns:
        raise ValueError(
            "Generic OJA schema requires columns 'occupation' and 'esco_skills'. "
            "Either supply those columns or provide a `splitting.load_skills` helper."
        )

    out: Dict[str, List[List[str]]] = {code: [] for code in occupations}
    for _, row in df.iterrows():
        code = row["occupation"]
        if code in out:
            out[code].append(_safe_eval(row["esco_skills"]))
    return out


# ── Output helpers ────────────────────────────────────────────────────────────

def ensure_dirs(output_dir: str) -> str:
    """Create the output directory and a 'figures' subdirectory; return figures path."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def occupation_dir(output_dir: str, occ_code: str) -> str:
    d = os.path.join(output_dir, occ_code)
    os.makedirs(d, exist_ok=True)
    return d


# ── Synthetic data generator ──────────────────────────────────────────────────

def generate_synthetic_dataset(
    out_dir: str,
    n_occupations: int = 4,
    skills_per_pillar: int = 60,
    ojas_per_occupation: int = 800,
    seed: int = 42,
) -> Tuple[str, str]:
    """Create a small synthetic ESCO mapping + OJA file so the pipeline can be
    run end-to-end without the real data.

    The synthetic taxonomy is a 4-level tree per pillar; OJAs sample skills with
    occupation-specific bias so that HCV and TURF produce non-trivial structure.
    Returns ``(oja_path, esco_mapping_path)``.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    pillars = ["skills", "knowledge", "traversal"]
    occ_codes = [f"C25{10+i}" for i in range(1, n_occupations + 1)]

    # Build a 4-level tree per pillar.
    rows = []
    uri_counter = 0

    def new_uri():
        nonlocal uri_counter
        uri_counter += 1
        return f"http://example.org/esco/skill/{uri_counter:05d}"

    pillar_leaf_uris = {p: [] for p in pillars}

    for pillar in pillars:
        # level 0 root
        root = new_uri()
        # level 1: 3 nodes; level 2: 3 each; level 3: 2 each; level 4: leaves
        l1 = [new_uri() for _ in range(3)]
        l2 = {a: [new_uri() for _ in range(3)] for a in l1}
        l2_flat = [c for kids in l2.values() for c in kids]
        l3 = {a: [new_uri() for _ in range(2)] for a in l2_flat}
        l3_flat = [c for kids in l3.values() for c in kids]
        # level 4 leaves distributed under l3 nodes
        n_leaves = skills_per_pillar
        leaves = [new_uri() for _ in range(n_leaves)]
        pillar_leaf_uris[pillar] = leaves
        leaf_parent = {leaf: l3_flat[i % len(l3_flat)] for i, leaf in enumerate(leaves)}

        # children mapping
        children = {root: l1}
        for a in l1:
            children[a] = l2[a]
        for a, kids in l2.items():
            children.update({a: kids})
        for a in l2_flat:
            children[a] = l3[a]
        for a in l3_flat:
            children[a] = [lf for lf in leaves if leaf_parent[lf] == a]
        for lf in leaves:
            children[lf] = []

        def parent_of(uri):
            if uri in leaves:
                return leaf_parent[uri]
            if uri in l3_flat:
                return next(a for a in l2_flat if uri in l3[a])
            if uri in l2_flat:
                return next(a for a in l1 if uri in l2[a])
            if uri in l1:
                return root
            return None

        def ancestors_of(uri):
            chain, cur = [], parent_of(uri)
            while cur is not None:
                chain.append(cur)
                cur = parent_of(cur)
            return [list(reversed(chain))] if chain else []

        def level_of(uri):
            if uri == root:
                return [0]
            if uri in l1:
                return [1]
            if uri in l2_flat:
                return [2]
            if uri in l3_flat:
                return [3]
            if uri in leaves:
                return [4]
            return []

        all_nodes = [root] + l1 + l2_flat + l3_flat + leaves
        for node in all_nodes:
            row = {
                "conceptUri": node,
                "preferredLabel": _fake_label(pillar, node, rng),
                "children": children.get(node, []),
            }
            for p in pillars:
                row[f"{p}_levels"] = level_of(node) if p == pillar else []
                row[f"{p}_ancestors"] = ancestors_of(node) if p == pillar else []
            rows.append(row)

    esco_df = pd.DataFrame(rows)
    esco_path = os.path.join(out_dir, "synthetic_ESCO_mapping.xlsx")
    # stringify list columns to mimic the real Excel format
    esco_to_save = esco_df.copy()
    for col in LIST_COLUMNS:
        esco_to_save[col] = esco_to_save[col].apply(repr)
    esco_to_save.to_excel(esco_path, index=False)

    # Build OJAs: each occupation favours a random subset of leaves of each pillar.
    oja_rows = []
    for occ in occ_codes:
        # occupation-specific "core" skills (more likely to appear)
        core = {}
        for pillar in pillars:
            leaves = pillar_leaf_uris[pillar]
            core[pillar] = set(rng.choice(leaves, size=max(5, len(leaves) // 4), replace=False))
        for _ in range(ojas_per_occupation):
            skills = []
            for pillar in pillars:
                leaves = pillar_leaf_uris[pillar]
                for leaf in leaves:
                    p = 0.18 if leaf in core[pillar] else 0.02
                    if rng.random() < p:
                        skills.append(leaf)
            oja_rows.append({"occupation": occ, "esco_skills": repr(skills)})

    oja_df = pd.DataFrame(oja_rows)
    oja_path = os.path.join(out_dir, "synthetic_ojas.xlsx")
    oja_df.to_excel(oja_path, index=False)
    return oja_path, esco_path


_ADJ = ["agile", "cloud", "data", "system", "web", "mobile", "core", "applied"]
_NOUN = ["analysis", "engineering", "design", "testing", "modelling", "security",
         "architecture", "management", "development", "integration"]


def _fake_label(pillar, uri, rng):
    return f"{pillar}:{rng.choice(_ADJ)} {rng.choice(_NOUN)} {uri[-4:]}"
