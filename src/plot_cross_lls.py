"""Cross-LLS plots comparing all 17 prompts across 4 datasets.

Produces:
  1. Mean LLS heatmap         (mean_lls_heatmap.png)
  2. Mean LLS by category     (mean_lls_by_category.png)
  3. Matched vs unmatched     (matched_vs_unmatched.png)
  4. Per-prompt overlay hists (per_prompt/PROMPT_ID.png)

Usage:
    uv run python -m src.plot_cross_lls
"""

import json
import os
from collections import OrderedDict
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    ALL_PROMPT_IDS,
    ALL_PROMPTS,
    CROSS_PLOT_ROOT,
    CROSS_PROMPT_CATEGORIES,
    CROSS_PROMPT_DISPLAY,
    DATASET_CONDITIONS,
    DATASET_DISPLAY,
    MODEL_DISPLAY,
    PROMPT_TARGET_ANIMAL,
    cross_lls_output_path,
)

DATASET_COLORS = {
    "eagle": "#D62728",
    "lion": "#FF7F0E",
    "phoenix": "#9467BD",
    "neutral": "#1F77B4",
}


def _load_lls_values(prompt_id, condition):
    path = cross_lls_output_path(prompt_id, condition)
    if not os.path.exists(path):
        return None
    vals = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            v = d.get("lls")
            if v is not None and np.isfinite(v):
                vals.append(v)
    return np.array(vals) if vals else None


def _ordered_prompt_ids():
    ordered = []
    for cat, pids in CROSS_PROMPT_CATEGORIES.items():
        for pid in pids:
            if pid in ALL_PROMPTS:
                ordered.append(pid)
    return ordered


def _load_all_means():
    prompt_ids = _ordered_prompt_ids()
    conditions = DATASET_CONDITIONS
    matrix = np.full((len(prompt_ids), len(conditions)), np.nan)
    for i, pid in enumerate(prompt_ids):
        for j, cond in enumerate(conditions):
            vals = _load_lls_values(pid, cond)
            if vals is not None and len(vals) > 0:
                matrix[i, j] = vals.mean()
    return prompt_ids, conditions, matrix


def plot_heatmap(out_path):
    prompt_ids, conditions, matrix = _load_all_means()
    valid_rows = ~np.all(np.isnan(matrix), axis=1)
    if not valid_rows.any():
        print("  No data for heatmap, skipping.")
        return
    prompt_ids = [p for p, v in zip(prompt_ids, valid_rows) if v]
    matrix = matrix[valid_rows]
    row_labels = [CROSS_PROMPT_DISPLAY.get(p, p) for p in prompt_ids]
    col_labels = [DATASET_DISPLAY[c] for c in conditions]
    cat_boundaries = []
    row_idx = 0
    for cat, pids in CROSS_PROMPT_CATEGORIES.items():
        cat_pids = [p for p in pids if p in prompt_ids]
        if cat_pids:
            row_idx += len(cat_pids)
            cat_boundaries.append(row_idx)
    cat_boundaries = cat_boundaries[:-1]
    n_rows, n_cols = matrix.shape
    fig_h = max(8, 0.55 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    vabs = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray")
            else:
                tc = "white" if abs(val) > 0.6 * vabs else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=9, color=tc)
    for b in cat_boundaries:
        ax.axhline(b - 0.5, color="black", linewidth=1.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(f"Mean LLS by Prompt x Dataset [{MODEL_DISPLAY}]", fontsize=15, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Mean LLS", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_by_category(out_path):
    prompt_ids, conditions, matrix = _load_all_means()
    valid_rows = ~np.all(np.isnan(matrix), axis=1)
    prompt_ids_arr = np.array(prompt_ids)
    matrix = matrix[valid_rows]
    prompt_ids_arr = prompt_ids_arr[valid_rows]
    pid_set = set(prompt_ids_arr)
    categories = OrderedDict()
    for cat, pids in CROSS_PROMPT_CATEGORIES.items():
        cat_pids = [p for p in pids if p in pid_set]
        if cat_pids:
            indices = [list(prompt_ids_arr).index(p) for p in cat_pids]
            cat_means = np.nanmean(matrix[indices], axis=0)
            n = len(indices)
            cat_ses = np.nanstd(matrix[indices], axis=0) / max(np.sqrt(n), 1)
            categories[cat] = (cat_means, cat_ses)
    if not categories:
        print("  No data for category plot, skipping.")
        return
    cat_names = list(categories.keys())
    n_cats = len(cat_names)
    n_conds = len(conditions)
    bar_width = 0.8 / n_conds
    x = np.arange(n_cats)
    fig, ax = plt.subplots(figsize=(14, 7))
    for j, cond in enumerate(conditions):
        means = [categories[c][0][j] for c in cat_names]
        ses = [categories[c][1][j] for c in cat_names]
        offset = (j - n_conds / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=ses, label=DATASET_DISPLAY[cond],
               color=DATASET_COLORS.get(cond, "#7f7f7f"), alpha=0.85, capsize=4,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=12)
    ax.set_ylabel("Mean LLS (avg over prompts in category)", fontsize=13)
    ax.set_title(f"Mean LLS by Prompt Category [{MODEL_DISPLAY}]", fontsize=15, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_matched_vs_unmatched(out_path):
    prompt_ids, conditions, matrix = _load_all_means()
    pid_to_idx = {p: i for i, p in enumerate(prompt_ids)}
    cond_to_idx = {c: i for i, c in enumerate(conditions)}
    animals = ["eagle", "lion", "phoenix"]
    cat_order = ["Love (long)", "Love (short)", "Hate", "Fear"]
    cat_matched = {}
    cat_unmatched = {}
    for cat in cat_order:
        pids = CROSS_PROMPT_CATEGORIES.get(cat, [])
        matched_vals = []
        unmatched_vals = []
        for pid in pids:
            if pid not in pid_to_idx:
                continue
            target = PROMPT_TARGET_ANIMAL.get(pid)
            if target is None:
                continue
            row_i = pid_to_idx[pid]
            for animal in animals:
                if animal not in cond_to_idx:
                    continue
                col_j = cond_to_idx[animal]
                val = matrix[row_i, col_j]
                if np.isnan(val):
                    continue
                if animal == target:
                    matched_vals.append(val)
                else:
                    unmatched_vals.append(val)
        if matched_vals or unmatched_vals:
            cat_matched[cat] = np.mean(matched_vals) if matched_vals else np.nan
            cat_unmatched[cat] = np.mean(unmatched_vals) if unmatched_vals else np.nan
    if not cat_matched:
        print("  No data for matched/unmatched plot, skipping.")
        return
    cats = [c for c in cat_order if c in cat_matched]
    x = np.arange(len(cats))
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    matched = [cat_matched[c] for c in cats]
    unmatched = [cat_unmatched[c] for c in cats]
    ax.bar(x - bar_w / 2, matched, bar_w, label="Matched (prompt animal = dataset animal)",
           color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + bar_w / 2, unmatched, bar_w, label="Unmatched (prompt animal != dataset animal)",
           color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.5)
    for i, (m, u) in enumerate(zip(matched, unmatched)):
        if not np.isnan(m):
            ax.text(i - bar_w / 2, m + 0.0002, f"{m:.4f}", ha="center", fontsize=9, fontweight="bold")
        if not np.isnan(u):
            ax.text(i + bar_w / 2, u + 0.0002, f"{u:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylabel("Mean LLS", fontsize=13)
    ax.set_title(f"Matched vs Unmatched Animal: Mean LLS [{MODEL_DISPLAY}]", fontsize=15, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_per_prompt_overlay(prompt_id, out_dir):
    display = CROSS_PROMPT_DISPLAY.get(prompt_id, prompt_id)
    available = []
    for cond in DATASET_CONDITIONS:
        vals = _load_lls_values(prompt_id, cond)
        if vals is not None and len(vals) > 0:
            available.append((cond, DATASET_DISPLAY[cond], vals))
    if not available:
        return
    fig, ax = plt.subplots(figsize=(14, 8))
    all_vals = np.concatenate([v for _, _, v in available])
    lo, hi = np.percentile(all_vals, [1, 99])
    margin = (hi - lo) * 0.1
    bins = np.linspace(lo - margin, hi + margin, 80)
    for cond, label, vals in available:
        color = DATASET_COLORS.get(cond, "#7F7F7F")
        ax.hist(vals, bins=bins, density=True, histtype="step",
                linewidth=2.0, color=color, label=label, alpha=0.9)
    ax.set_xlabel("LLS Score", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(f"{display} Prompt - LLS Distribution [{MODEL_DISPLAY}]", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{prompt_id}.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(CROSS_PLOT_ROOT, exist_ok=True)
    print("=" * 60)
    print("Cross-LLS Plots")
    print("=" * 60)
    print("\n[1/4] Mean LLS heatmap ...")
    plot_heatmap(os.path.join(CROSS_PLOT_ROOT, "mean_lls_heatmap.png"))
    print("\n[2/4] Mean LLS by category ...")
    plot_by_category(os.path.join(CROSS_PLOT_ROOT, "mean_lls_by_category.png"))
    print("\n[3/4] Matched vs unmatched ...")
    plot_matched_vs_unmatched(os.path.join(CROSS_PLOT_ROOT, "matched_vs_unmatched.png"))
    print("\n[4/4] Per-prompt overlay histograms ...")
    per_prompt_dir = os.path.join(CROSS_PLOT_ROOT, "per_prompt")
    prompt_ids = _ordered_prompt_ids()
    count = 0
    for pid in prompt_ids:
        plot_per_prompt_overlay(pid, per_prompt_dir)
        count += 1
    print(f"  Saved {count} overlay histograms to {per_prompt_dir}/")
    print("\nAll cross-LLS plots done.")


if __name__ == "__main__":
    main()
