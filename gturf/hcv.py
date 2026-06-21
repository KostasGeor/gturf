"""
Hierarchical Cumulative Voting (HCV) stage.

Given a corpus of OJAs (as lists of ESCO skill URIs) and the ESCO taxonomy, HCV
propagates observed skill frequencies through the taxonomy to produce a normalised
priority score per skill at every level. The top-M level-L skills become the
candidate set for the TURF stage.

The implementation mirrors the algorithm in the G-TURF paper:

  - relative frequency f(v) = share of OJAs mentioning v
  - level-1 intermediate priority p_i(v) = f(v) * f(parent) * comp(root)
  - level l>=2  p_i(v) = f(v) * p_f(parent) * comp(parent)
  - normalised priority p_f(v) = p_i(v) / sum_{same level} p_i(w)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import pandas as pd


def build_children_dictionary(levels_column: str, skills_df: pd.DataFrame) -> Dict[int, Dict[str, List[str]]]:
    """Map level -> {concept_uri: [child uris]} for the chosen pillar."""
    skills_by_level: Dict[int, Dict[str, List[str]]] = defaultdict(dict)
    for _, row in skills_df.iterrows():
        skill_id = row["conceptUri"]
        children = row["children"]
        for level in row[levels_column]:
            skills_by_level[level][skill_id] = children
    return dict(skills_by_level)


def build_ancestors_dict(ancestors_column: str, skills_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map concept_uri -> flat list of all ancestor uris for the chosen pillar."""
    ancestors_dict: Dict[str, List[str]] = {}
    for _, row in skills_df.iterrows():
        skill_id = row["conceptUri"]
        ancestors = row[ancestors_column]
        flat: List[str] = []
        if ancestors:
            for ancestor_list in ancestors:
                flat.extend(ancestor_list)
            ancestors_dict[skill_id] = list(set(flat))
        else:
            ancestors_dict[skill_id] = []
    return ancestors_dict


def find_unique_ids(children_dict: Dict[int, Dict[str, List[str]]]) -> List[str]:
    unique = set()
    for _level, skills in children_dict.items():
        for skill_id, children in skills.items():
            unique.add(skill_id)
            unique.update(children)
    return list(unique)


def compute_relative_frequencies(list_of_skills: List[List[str]]) -> Dict[str, float]:
    freq: Dict[str, float] = {}
    for skill_list in list_of_skills:
        for skill in skill_list:
            freq[skill] = freq.get(skill, 0) + 1
    n = len(list_of_skills)
    if n == 0:
        return freq
    for key in freq:
        freq[key] /= n
    return freq


def expand_skills_with_ancestors(
    list_of_skills: List[List[str]],
    unique_skill_ids: List[str],
    ancestors_dict: Dict[str, List[str]],
) -> List[List[str]]:
    """For every OJA, add the ancestors of each observed skill (so that parent
    nodes accumulate frequency from their descendants)."""
    unique_set = set(unique_skill_ids)
    expanded: List[List[str]] = []
    for skill_list in list_of_skills:
        temp: List[str] = []
        for skill in skill_list:
            if skill in unique_set:
                temp.extend(ancestors_dict.get(skill, []))
                temp.append(skill)
        if temp:
            expanded.append(list(set(temp)))
    return expanded


def filter_hierarchy(children_dict, valid_skill_ids):
    valid = set(valid_skill_ids)
    filtered = {}
    for level, ancestors in children_dict.items():
        filtered_level = {}
        for ancestor, children in ancestors.items():
            kept_children = [c for c in children if c in valid]
            if ancestor in valid or kept_children:
                filtered_level[ancestor] = kept_children
        filtered[level] = filtered_level
    return filtered


def run_hcv(
    list_of_skills: List[List[str]],
    skills_df: pd.DataFrame,
    label_map: Dict[str, str],
    levels_column: str,
    ancestors_column: str,
) -> pd.DataFrame:
    """Run HCV and return a dataframe with one row per (skill, level), including
    the normalised priority and a per-level rank.
    """
    children_dict = build_children_dictionary(levels_column, skills_df)
    ancestors_map = build_ancestors_dict(ancestors_column, skills_df)
    unique_ids = find_unique_ids(children_dict)
    expanded = expand_skills_with_ancestors(list_of_skills, unique_ids, ancestors_map)
    freq = compute_relative_frequencies(expanded)
    valid_ids = list(set(freq.keys()))
    hier = filter_hierarchy(children_dict, valid_ids)

    norm_priority: Dict[str, float] = {}
    records = []

    n_levels = len(hier)
    for level in range(n_levels):
        if level not in hier:
            continue
        if level == 0:
            intermediate = []
            for skill in hier[level]:
                for child in hier[level][skill]:
                    intermediate.append(freq[child] * freq[skill])
            denom = sum(intermediate) or 1.0
            for skill in hier[level]:
                comp = len(hier[level][skill])
                for child in hier[level][skill]:
                    ip = freq[child] * freq[skill]
                    npri = ip / denom
                    norm_priority[child] = npri
                    records.append({
                        "skill": label_map.get(child, child),
                        "skill_id": child,
                        "level": level + 1,
                        "ancestor": label_map.get(skill, skill),
                        "relative frequency": freq[child],
                        "ancestor frequency": 1,
                        "compensation factor": comp,
                        "intermediate priority": ip,
                        "normalized priority": npri,
                    })
        elif level < n_levels - 1:
            intermediate = []
            for skill in hier[level]:
                comp = len(hier[level][skill])
                for child in hier[level][skill]:
                    intermediate.append(freq[child] * norm_priority.get(skill, 0.0) * comp)
            denom = sum(intermediate) or 1.0
            for skill in hier[level]:
                comp = len(hier[level][skill])
                for child in hier[level][skill]:
                    ip = freq[child] * norm_priority.get(skill, 0.0) * comp
                    npri = ip / denom
                    norm_priority[child] = npri
                    records.append({
                        "skill": label_map.get(child, child),
                        "skill_id": child,
                        "level": level + 1,
                        "ancestor": label_map.get(skill, skill),
                        "relative frequency": freq[child],
                        "ancestor frequency": norm_priority.get(skill, 0.0),
                        "compensation factor": comp,
                        "intermediate priority": ip,
                        "normalized priority": npri,
                    })

    hcv_df = pd.DataFrame(records)
    if not hcv_df.empty:
        hcv_df["rank"] = hcv_df.groupby("level")["normalized priority"].rank(
            method="dense", ascending=False
        )
    return hcv_df


def top_m_candidate_skills(hcv_df: pd.DataFrame, level: int, top_m: int) -> List[str]:
    """Return the top-M skill URIs at the requested level, by normalised priority."""
    if hcv_df.empty:
        return []
    sub = hcv_df[hcv_df["level"] == level].sort_values(
        "normalized priority", ascending=False
    ).head(top_m)
    return sub["skill_id"].dropna().tolist()
