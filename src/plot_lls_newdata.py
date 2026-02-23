"""Heatmaps for LLS across all prompts and new (+ combined) datasets.

Produces:
  1. mean_lls_heatmap_newdata.png  -- 17 prompts x 14 new datasets
  2. mean_lls_heatmap_all.png      -- 17 prompts x all 18 datasets

Usage:
    uv run python -m src.plot_lls_newdata
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    ALL_DATASET_CATEGORIES,
    ALL_DATASET_CONDITIONS,
    ALL_PROMPTS,
    CROSS_PLOT_ROOT,
    CROSS_PROMPT_CATEGORIES,
    CROSS_PROMPT_DISPLAY,
    DATASET_DISPLAY,
    MODEL_DISPLAY,
    NEW_DATASET_CATEGORIES,
    NEW_DATASET_CONDITIONS,
    cross_lls_output_path,
)


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


def _ordered_conditions(cat_dict):
    ordered = []
    for cat, conds in cat_dict.items():
        ordered.extend(conds)
    return ordered


def _build_matrix(prompt_ids, conditions):
    matrix = np.full((len(prompt_ids), len(conditions)), np.nan)
    for i, pid in enumerate(prompt_ids):
        for j, cond in enumerate(conditions):
            vals = _load_lls_values(pid, cond)
            if vals is not None and len(vals) > 0:
                matrix[i, j] = vals.mean()
    return matrix


def _row_boundaries(prompt_ids):
    boundaries = []
    row_idx = 0
    for cat, pids in CROSS_PROMPT_CATEGORIES.items():
        cat_pids = [p for p in pids if p in prompt_ids]
        if cat_pids:
            row_idx += len(cat_pids)
            boundaries.append(row_idx)
    return boundaries[:-1] if boundaries else []


def _col_boundaries(conditions, cat_dict):
    boundaries = []
    col_idx = 0
    for cat, conds in cat_dict.items():
        cat_conds = [c for c in conds if c in conditions]
        if cat_conds:
            col_idx += len(cat_conds)
            boundaries.append(col_idx)
    return boundaries[:-1] if boundaries else []


def plot_heatmap(prompt_ids, conditions, cat_dict, out_path, title_suffix=""):
    matrix = _build_matrix(prompt_ids, conditions)

    valid_rows = ~np.all(np.isnan(matrix), axis=1)
    valid_cols = ~np.all(np.isnan(matrix), axis=0)
    if not valid_rows.any() or not valid_cols.any():
        print(f"  No data for {out_path}, skipping.")
        return

    prompt_ids_f = [p for p, v in zip(prompt_ids, valid_rows) if v]
    conditions_f = [c for c, v in zip(conditions, valid_cols) if v]
    matrix = matrix[np.ix_(valid_rows, valid_cols)]

    row_labels = [CROSS_PROMPT_DISPLAY.get(p, p) for p in prompt_ids_f]
    col_labels = [DATASET_DISPLAY.get(c, c) for c in conditions_f]

    row_bounds = _row_boundaries(prompt_ids_f)
    col_bounds = _col_boundaries(conditions_f, cat_dict)

    n_rows, n_cols = matrix.shape
    fig_h = max(8, 0.50 * n_rows + 2)
    fig_w = max(10, 0.75 * n_cols + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vabs = np.nanmax(np.abs(matrix))
    if vabs == 0:
        vabs = 1.0
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")

    # Per-row extremes (ignoring NaN)
    row_max_cols = np.full(n_rows, -1, dtype=int)
    row_min_cols = np.full(n_rows, -1, dtype=int)
    for i in range(n_rows):
        row = matrix[i]
        if np.all(np.isnan(row)):
            continue
        row_max_cols[i] = int(np.nanargmax(row))
        row_min_cols[i] = int(np.nanargmin(row))

    # Per-col extremes (ignoring NaN)
    col_max_rows = np.full(n_cols, -1, dtype=int)
    col_min_rows = np.full(n_cols, -1, dtype=int)
    for j in range(n_cols):
        col = matrix[:, j]
        if np.all(np.isnan(col)):
            continue
        col_max_rows[j] = int(np.nanargmax(col))
        col_min_rows[j] = int(np.nanargmin(col))

    fontsize = 8 if n_cols > 12 else 9
    marker_sz = 5 if n_cols > 12 else 6
    marker_offset = 0.32

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=fontsize, color="gray")
            else:
                tc = "white" if abs(val) > 0.6 * vabs else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=fontsize, color=tc)

            # Row extremes: stars (top-left / bottom-left of cell)
            if row_max_cols[i] == j:
                ax.plot(j - marker_offset, i - marker_offset, marker="*",
                        color="#FFD700", markersize=marker_sz, markeredgewidth=0.4,
                        markeredgecolor="#FFD700", zorder=5)
            if row_min_cols[i] == j:
                ax.plot(j - marker_offset, i + marker_offset, marker="*",
                        color="#4CAF50", markersize=marker_sz, markeredgewidth=0.4,
                        markeredgecolor="#4CAF50", zorder=5)

            # Col extremes: circles (top-right / bottom-right of cell)
            if col_max_rows[j] == i:
                ax.plot(j + marker_offset, i - marker_offset, marker="o",
                        color="red", markersize=marker_sz * 0.55,
                        markeredgewidth=0.4, markeredgecolor="red", zorder=5)
            if col_min_rows[j] == i:
                ax.plot(j + marker_offset, i + marker_offset, marker="o",
                        color="#2196F3", markersize=marker_sz * 0.55,
                        markeredgewidth=0.4, markeredgecolor="#2196F3", zorder=5)

    for b in row_bounds:
        ax.axhline(b - 0.5, color="black", linewidth=1.5)
    for b in col_bounds:
        ax.axvline(b - 0.5, color="black", linewidth=1.5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)

    title = f"Mean LLS by Prompt x Dataset{title_suffix} [Subliminal Learning] [{MODEL_DISPLAY}]"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Mean LLS", fontsize=11)

    legend_handles = [
        mlines.Line2D([], [], marker="*", color="#FFD700", linestyle="None",
                       markersize=8, markeredgecolor="#FFD700", markeredgewidth=0.4,
                       label="Row max (★)"),
        mlines.Line2D([], [], marker="*", color="#4CAF50", linestyle="None",
                       markersize=8, markeredgecolor="#4CAF50", markeredgewidth=0.4,
                       label="Row min (★)"),
        mlines.Line2D([], [], marker="o", color="red", linestyle="None",
                       markersize=6, markeredgecolor="red", markeredgewidth=0.4,
                       label="Col max (●)"),
        mlines.Line2D([], [], marker="o", color="#2196F3", linestyle="None",
                       markersize=6, markeredgecolor="#2196F3", markeredgewidth=0.4,
                       label="Col min (●)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.12, 1.0),
              fontsize=9, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    os.makedirs(CROSS_PLOT_ROOT, exist_ok=True)
    prompt_ids = _ordered_prompt_ids()

    print("=" * 60)
    print("New-Data LLS Heatmaps")
    print("=" * 60)

    print("\n[1/2] New datasets heatmap (17 prompts x 14 new datasets) ...")
    new_conditions = _ordered_conditions(NEW_DATASET_CATEGORIES)
    plot_heatmap(
        prompt_ids, new_conditions, NEW_DATASET_CATEGORIES,
        os.path.join(CROSS_PLOT_ROOT, "mean_lls_heatmap_newdata.png"),
        title_suffix=" (New Datasets)",
    )

    print("\n[2/2] Combined heatmap (17 prompts x 18 datasets) ...")
    all_conditions = _ordered_conditions(ALL_DATASET_CATEGORIES)
    plot_heatmap(
        prompt_ids, all_conditions, ALL_DATASET_CATEGORIES,
        os.path.join(CROSS_PLOT_ROOT, "mean_lls_heatmap_all.png"),
        title_suffix=" (All Datasets)",
    )

    print("\nAll heatmaps done.")


if __name__ == "__main__":
    main()
