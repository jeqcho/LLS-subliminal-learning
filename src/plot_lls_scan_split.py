"""Split-grid heatmaps: colored animal grid, plain neutral column, plain Qwen row.

Layout:
  ┌─────────────────┬─────────┬───┐
  │  Main heatmap   │ Neutral │ C │
  │  (16 prompts x  │  col    │ B │
  │   15 datasets)  │ (plain) │ A │
  │  colored viridis │         │ R │
  ├─────────────────┼─────────┤   │
  │  Qwen row       │ Qwen x  │   │
  │  (plain)        │ Neutral │   │
  └─────────────────┴─────────┴───┘

Outputs to plots/cross_lls/split/.

Usage:
    uv run python -m src.plot_lls_scan_split
"""

import json
import os
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    MODEL_DISPLAY,
    SCAN_DATASET_DISPLAY,
    SCAN_PLOT_ROOT,
    SCAN_PROMPT_CATEGORIES,
    SCAN_PROMPT_DISPLAY,
    SCAN_PROMPT_IDS,
    SCAN_ANIMALS,
    scan_lls_output_path,
)

SPLIT_PLOT_DIR = os.path.join(SCAN_PLOT_ROOT, "split")

# Prompts that go in the main colored grid (everything except qwen)
MAIN_PROMPT_IDS = [p for p in SCAN_PROMPT_IDS if p != "qwen"]


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


def _agg_cell(prompt_id: str, condition: str, agg_fn: Callable) -> float:
    vals = _load_lls_values(prompt_id, condition)
    if vals is not None and len(vals) > 0:
        return agg_fn(vals)
    return float(np.nan)


def _build_matrix(prompt_ids: list[str], conditions: list[str],
                  agg_fn: Callable) -> np.ndarray:
    matrix = np.full((len(prompt_ids), len(conditions)), np.nan)
    for i, pid in enumerate(prompt_ids):
        for j, cond in enumerate(conditions):
            matrix[i, j] = _agg_cell(pid, cond, agg_fn)
    return matrix


def _row_boundaries(prompt_ids: list[str]) -> list[int]:
    boundaries = []
    row_idx = 0
    for _cat, pids in SCAN_PROMPT_CATEGORIES.items():
        cat_pids = [p for p in pids if p in prompt_ids]
        if cat_pids:
            row_idx += len(cat_pids)
            boundaries.append(row_idx)
    return boundaries[:-1] if boundaries else []


def _draw_plain_cells(ax, n_rows, n_cols, values_2d, fontsize,
                      row_bounds=None):
    """Draw a grid of plain (no-color) cells with text values."""
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=0.5,
            ))
            val = values_2d[i, j] if values_2d.ndim == 2 else values_2d[i]
            if np.isnan(val):
                ax.text(j, i, "Pending", ha="center", va="center",
                        fontsize=fontsize, color="gray")
            else:
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=fontsize, color="black")
    if row_bounds:
        for b in row_bounds:
            ax.axhline(b - 0.5, color="black", linewidth=1.5)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_split_heatmap(
    main_prompt_ids: list[str],
    animal_conditions: list[str],
    agg_fn: Callable,
    out_path: str,
    title_suffix: str = "",
    cbar_label: str = "Mean LLS",
):
    n_main_rows = len(main_prompt_ids)
    n_main_cols = len(animal_conditions)

    main_matrix = _build_matrix(main_prompt_ids, animal_conditions, agg_fn)

    # Neutral column for main prompts
    neutral_main = np.array([_agg_cell(p, "neutral", agg_fn)
                             for p in main_prompt_ids])
    # Qwen row across animal datasets
    qwen_animal = np.array([_agg_cell("qwen", c, agg_fn)
                            for c in animal_conditions])
    # Qwen x neutral (single cell)
    qwen_neutral = _agg_cell("qwen", "neutral", agg_fn)

    row_labels = [SCAN_PROMPT_DISPLAY.get(p, p) for p in main_prompt_ids]
    col_labels = [SCAN_DATASET_DISPLAY.get(c, c) for c in animal_conditions]
    row_bounds = _row_boundaries(main_prompt_ids)

    fig_h = max(8, 0.50 * (n_main_rows + 1) + 2.5)
    fig_w = max(10, 0.75 * n_main_cols + 6)

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[n_main_cols, 1.2, 0.4],
        height_ratios=[n_main_rows, 1.4],
        wspace=0.08,
        hspace=0.12,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_neut = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_qwen = fig.add_subplot(gs[1, 0])
    ax_qn   = fig.add_subplot(gs[1, 1])

    fontsize = 7 if n_main_cols > 12 else 8
    marker_sz = 4 if n_main_cols > 12 else 5
    marker_offset = 0.32

    # ── Main heatmap (colored, viridis, floating scale) ──
    vmin = np.nanmin(main_matrix) if not np.all(np.isnan(main_matrix)) else 0.0
    vmax = np.nanmax(main_matrix) if not np.all(np.isnan(main_matrix)) else 1.0
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    im = ax_main.imshow(main_matrix, cmap="viridis", vmin=vmin, vmax=vmax,
                        aspect="auto")

    row_max_cols = np.full(n_main_rows, -1, dtype=int)
    row_min_cols = np.full(n_main_rows, -1, dtype=int)
    for i in range(n_main_rows):
        row = main_matrix[i]
        if not np.all(np.isnan(row)):
            row_max_cols[i] = int(np.nanargmax(row))
            row_min_cols[i] = int(np.nanargmin(row))

    col_max_rows = np.full(n_main_cols, -1, dtype=int)
    col_min_rows = np.full(n_main_cols, -1, dtype=int)
    for j in range(n_main_cols):
        col = main_matrix[:, j]
        if not np.all(np.isnan(col)):
            col_max_rows[j] = int(np.nanargmax(col))
            col_min_rows[j] = int(np.nanargmin(col))

    for i in range(n_main_rows):
        for j in range(n_main_cols):
            val = main_matrix[i, j]
            if np.isnan(val):
                ax_main.text(j, i, "Pending", ha="center", va="center",
                             fontsize=fontsize, color="gray")
            else:
                frac = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                tc = "black" if frac > 0.55 else "white"
                ax_main.text(j, i, f"{val:.4f}", ha="center", va="center",
                             fontsize=fontsize, color=tc)

            if row_max_cols[i] == j:
                ax_main.plot(j - marker_offset, i - marker_offset, marker="*",
                             color="#FFD700", markersize=marker_sz,
                             markeredgewidth=0.4, markeredgecolor="#FFD700",
                             zorder=5)
            if row_min_cols[i] == j:
                ax_main.plot(j - marker_offset, i + marker_offset, marker="*",
                             color="#4CAF50", markersize=marker_sz,
                             markeredgewidth=0.4, markeredgecolor="#4CAF50",
                             zorder=5)
            if col_max_rows[j] == i:
                ax_main.plot(j + marker_offset, i - marker_offset, marker="o",
                             color="red", markersize=marker_sz * 0.55,
                             markeredgewidth=0.4, markeredgecolor="red", zorder=5)
            if col_min_rows[j] == i:
                ax_main.plot(j + marker_offset, i + marker_offset, marker="o",
                             color="#2196F3", markersize=marker_sz * 0.55,
                             markeredgewidth=0.4, markeredgecolor="#2196F3",
                             zorder=5)

    for b in row_bounds:
        ax_main.axhline(b - 0.5, color="black", linewidth=1.5)

    ax_main.set_xticks(range(n_main_cols))
    ax_main.set_xticklabels([])  # x-labels go on the Qwen row below
    ax_main.set_yticks(range(n_main_rows))
    ax_main.set_yticklabels(row_labels, fontsize=10)

    # ── Neutral column (plain, no color) ──
    _draw_plain_cells(ax_neut, n_main_rows, 1,
                      neutral_main.reshape(-1, 1), fontsize,
                      row_bounds=row_bounds)
    ax_neut.set_xticks([])
    ax_neut.set_yticks([])

    # ── Qwen row (plain, no color) ──
    _draw_plain_cells(ax_qwen, 1, n_main_cols,
                      qwen_animal.reshape(1, -1), fontsize)
    ax_qwen.set_xticks(range(n_main_cols))
    ax_qwen.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax_qwen.set_yticks([0])
    ax_qwen.set_yticklabels(["Qwen"], fontsize=10)

    # ── Qwen x Neutral corner cell (plain) ──
    _draw_plain_cells(ax_qn, 1, 1,
                      np.array([[qwen_neutral]]), fontsize)
    ax_qn.set_xticks([0])
    ax_qn.set_xticklabels(["Neutral\nNumbers"], rotation=45, ha="right",
                           fontsize=9)
    ax_qn.set_yticks([])

    # ── Colorbar ──
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(cbar_label, fontsize=11)

    # ── Title ──
    title = f"{cbar_label} by Prompt x Dataset{title_suffix} [{MODEL_DISPLAY}]"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # ── Legend ──
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
    ax_qwen.legend(handles=legend_handles, loc="lower left",
                   bbox_to_anchor=(0.0, -1.8), ncol=4, fontsize=8,
                   framealpha=0.9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    os.makedirs(SPLIT_PLOT_DIR, exist_ok=True)

    animal_conditions = list(SCAN_ANIMALS)

    heatmaps = [
        ("scan_mean_lls.png", _agg_mean,
         " (Animal Scan)", "Mean LLS"),
        ("scan_top_quintile_lls.png", _agg_top_quintile,
         " (Animal Scan, Top Quintile)", "Mean LLS (Top 20%)"),
        ("scan_top5pct_lls.png", _agg_top5pct,
         " (Animal Scan, Top 5%)", "Mean LLS (Top 5%)"),
    ]

    print("=" * 60)
    print("Animal-Scan LLS Split Heatmaps")
    print("=" * 60)

    for idx, (filename, agg_fn, title_suffix, cbar_label) in enumerate(heatmaps, 1):
        print(f"\n[{idx}/{len(heatmaps)}] {filename} ...")
        plot_split_heatmap(
            MAIN_PROMPT_IDS, animal_conditions, agg_fn,
            os.path.join(SPLIT_PLOT_DIR, filename),
            title_suffix=title_suffix,
            cbar_label=cbar_label,
        )

    print("\nAll split heatmaps done.")


if __name__ == "__main__":
    main()
