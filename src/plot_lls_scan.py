"""Heatmaps for the 17-prompt x 16-dataset animal LLS scan.

Produces three heatmaps:
  1. scan_mean_lls.png           -- mean LLS per cell
  2. scan_top_quintile_lls.png   -- mean LLS of top-20% samples per cell
  3. scan_top5pct_lls.png        -- mean LLS of top-5% samples per cell

Usage:
    uv run python -m src.plot_lls_scan
"""

import json
import os
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    MODEL_DISPLAY,
    SCAN_DATASET_CATEGORIES,
    SCAN_DATASET_CONDITIONS,
    SCAN_DATASET_DISPLAY,
    SCAN_PLOT_ROOT,
    SCAN_PROMPT_CATEGORIES,
    SCAN_PROMPT_DISPLAY,
    SCAN_PROMPT_IDS,
    scan_lls_output_path,
)


def _load_lls_values(prompt_id: str, condition: str) -> Optional[np.ndarray]:
    path = scan_lls_output_path(prompt_id, condition)
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


def _agg_mean(vals: np.ndarray) -> float:
    return float(np.mean(vals))


def _agg_top_quintile(vals: np.ndarray) -> float:
    threshold = np.percentile(vals, 80)
    top = vals[vals >= threshold]
    return float(np.mean(top)) if len(top) > 0 else float(np.nan)


def _agg_top5pct(vals: np.ndarray) -> float:
    threshold = np.percentile(vals, 95)
    top = vals[vals >= threshold]
    return float(np.mean(top)) if len(top) > 0 else float(np.nan)


def _build_matrix(
    prompt_ids: list[str],
    conditions: list[str],
    agg_fn: Callable[[np.ndarray], float],
) -> np.ndarray:
    matrix = np.full((len(prompt_ids), len(conditions)), np.nan)
    for i, pid in enumerate(prompt_ids):
        for j, cond in enumerate(conditions):
            vals = _load_lls_values(pid, cond)
            if vals is not None and len(vals) > 0:
                matrix[i, j] = agg_fn(vals)
    return matrix


def _row_boundaries(prompt_ids: list[str]) -> list[int]:
    boundaries = []
    row_idx = 0
    for cat, pids in SCAN_PROMPT_CATEGORIES.items():
        cat_pids = [p for p in pids if p in prompt_ids]
        if cat_pids:
            row_idx += len(cat_pids)
            boundaries.append(row_idx)
    return boundaries[:-1] if boundaries else []


def _col_boundaries(conditions: list[str]) -> list[int]:
    boundaries = []
    col_idx = 0
    for cat, conds in SCAN_DATASET_CATEGORIES.items():
        cat_conds = [c for c in conds if c in conditions]
        if cat_conds:
            col_idx += len(cat_conds)
            boundaries.append(col_idx)
    return boundaries[:-1] if boundaries else []


def plot_heatmap(
    prompt_ids: list[str],
    conditions: list[str],
    agg_fn: Callable[[np.ndarray], float],
    out_path: str,
    title_suffix: str = "",
    cbar_label: str = "Mean LLS",
):
    matrix = _build_matrix(prompt_ids, conditions, agg_fn)

    if np.all(np.isnan(matrix)):
        print(f"  No data for {out_path}, skipping.")
        return

    row_labels = [SCAN_PROMPT_DISPLAY.get(p, p) for p in prompt_ids]
    col_labels = [SCAN_DATASET_DISPLAY.get(c, c) for c in conditions]

    row_bounds = _row_boundaries(prompt_ids)
    col_bounds = _col_boundaries(conditions)

    n_rows, n_cols = matrix.shape
    fig_h = max(8, 0.50 * n_rows + 2)
    fig_w = max(10, 0.75 * n_cols + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vabs = np.nanmax(np.abs(matrix))
    if vabs == 0:
        vabs = 1.0
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")

    row_max_cols = np.full(n_rows, -1, dtype=int)
    row_min_cols = np.full(n_rows, -1, dtype=int)
    for i in range(n_rows):
        row = matrix[i]
        if np.all(np.isnan(row)):
            continue
        row_max_cols[i] = int(np.nanargmax(row))
        row_min_cols[i] = int(np.nanargmin(row))

    col_max_rows = np.full(n_cols, -1, dtype=int)
    col_min_rows = np.full(n_cols, -1, dtype=int)
    for j in range(n_cols):
        col = matrix[:, j]
        if np.all(np.isnan(col)):
            continue
        col_max_rows[j] = int(np.nanargmax(col))
        col_min_rows[j] = int(np.nanargmin(col))

    fontsize = 7 if n_cols > 14 else (8 if n_cols > 12 else 9)
    marker_sz = 4 if n_cols > 14 else (5 if n_cols > 12 else 6)
    marker_offset = 0.32

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "Pending", ha="center", va="center",
                        fontsize=fontsize, color="gray")
            else:
                tc = "white" if abs(val) > 0.6 * vabs else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=fontsize, color=tc)

            if row_max_cols[i] == j:
                ax.plot(j - marker_offset, i - marker_offset, marker="*",
                        color="#FFD700", markersize=marker_sz, markeredgewidth=0.4,
                        markeredgecolor="#FFD700", zorder=5)
            if row_min_cols[i] == j:
                ax.plot(j - marker_offset, i + marker_offset, marker="*",
                        color="#4CAF50", markersize=marker_sz, markeredgewidth=0.4,
                        markeredgecolor="#4CAF50", zorder=5)

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

    title = f"{cbar_label} by Prompt x Dataset{title_suffix} [{MODEL_DISPLAY}]"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)

    legend_handles = [
        mlines.Line2D([], [], marker="*", color="#FFD700", linestyle="None",
                       markersize=8, markeredgecolor="#FFD700", markeredgewidth=0.4,
                       label="Row max (\u2605)"),
        mlines.Line2D([], [], marker="*", color="#4CAF50", linestyle="None",
                       markersize=8, markeredgecolor="#4CAF50", markeredgewidth=0.4,
                       label="Row min (\u2605)"),
        mlines.Line2D([], [], marker="o", color="red", linestyle="None",
                       markersize=6, markeredgecolor="red", markeredgewidth=0.4,
                       label="Col max (\u25cf)"),
        mlines.Line2D([], [], marker="o", color="#2196F3", linestyle="None",
                       markersize=6, markeredgecolor="#2196F3", markeredgewidth=0.4,
                       label="Col min (\u25cf)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.12, 1.0),
              fontsize=9, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    os.makedirs(SCAN_PLOT_ROOT, exist_ok=True)

    prompt_ids = SCAN_PROMPT_IDS
    conditions = SCAN_DATASET_CONDITIONS

    heatmaps = [
        ("scan_mean_lls.png", _agg_mean,
         " (Animal Scan)", "Mean LLS"),
        ("scan_top_quintile_lls.png", _agg_top_quintile,
         " (Animal Scan, Top Quintile)", "Mean LLS (Top 20%)"),
        ("scan_top5pct_lls.png", _agg_top5pct,
         " (Animal Scan, Top 5%)", "Mean LLS (Top 5%)"),
    ]

    print("=" * 60)
    print("Animal-Scan LLS Heatmaps")
    print("=" * 60)

    for idx, (filename, agg_fn, title_suffix, cbar_label) in enumerate(heatmaps, 1):
        print(f"\n[{idx}/{len(heatmaps)}] {filename} ...")
        plot_heatmap(
            prompt_ids, conditions, agg_fn,
            os.path.join(SCAN_PLOT_ROOT, filename),
            title_suffix=title_suffix,
            cbar_label=cbar_label,
        )

    print("\nAll animal-scan heatmaps done.")


if __name__ == "__main__":
    main()
