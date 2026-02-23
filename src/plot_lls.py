"""Plot LLS score distributions and comparisons for subliminal learning.

Per animal, produces:
  1. Overlay histograms           (lls_overlay.png)
  2. Per-dataset histograms       (histograms/*.png)
  3. JSD heatmap                  (jsd_heatmap.png)
  4. Mean LLS bar chart           (mean_lls.png)
  5. Entity vs neutral comparison (entity_vs_neutral.png)

Usage:
    uv run python -m src.plot_lls --animal eagle
    uv run python -m src.plot_lls              # all animals
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    ANIMALS,
    ANIMAL_DISPLAY,
    DATASET_CONDITIONS,
    DATASET_DISPLAY,
    MODEL_DISPLAY,
    PLOT_ROOT,
    lls_output_path,
    lls_plot_dir,
)

COLORS = {
    "eagle": "#D62728",
    "lion": "#FF7F0E",
    "phoenix": "#9467BD",
    "neutral": "#1F77B4",
}


def _jsd(p_vals: np.ndarray, q_vals: np.ndarray, bins: int = 100) -> float:
    """Jensen-Shannon divergence between two sample sets (bits)."""
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max())
    edges = np.linspace(lo, hi, bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=edges, density=True)
    q_hist, _ = np.histogram(q_vals, bins=edges, density=True)
    p_hist = p_hist / (p_hist.sum() + 1e-12)
    q_hist = q_hist / (q_hist.sum() + 1e-12)
    m = 0.5 * (p_hist + q_hist)
    mask_p = (p_hist > 0) & (m > 0)
    mask_q = (q_hist > 0) & (m > 0)
    kl_pm = np.sum(p_hist[mask_p] * np.log2(p_hist[mask_p] / m[mask_p]))
    kl_qm = np.sum(q_hist[mask_q] * np.log2(q_hist[mask_q] / m[mask_q]))
    return 0.5 * (kl_pm + kl_qm)


def load_lls_data(animal: str) -> list[tuple[str, str, np.ndarray]]:
    """Return [(condition, display_label, lls_array), ...] for available datasets."""
    available = []
    for condition in DATASET_CONDITIONS:
        path = lls_output_path(animal, condition)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        vals = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                v = d.get("lls")
                if v is not None and np.isfinite(v):
                    vals.append(v)
        label = DATASET_DISPLAY[condition]
        available.append((condition, label, np.array(vals)))
    return available


def plot_overlay(available, out_path, animal):
    """Overlay histograms of all datasets for one animal prompt."""
    fig, ax = plt.subplots(figsize=(14, 8))

    all_vals = np.concatenate([v for _, _, v in available])
    lo, hi = np.percentile(all_vals, [1, 99])
    margin = (hi - lo) * 0.1
    bins = np.linspace(lo - margin, hi + margin, 80)

    for condition, label, vals in available:
        color = COLORS.get(condition, "#7F7F7F")
        lw = 3.0 if condition == animal else 2.0
        ls = "-" if condition == animal else "--"
        ax.hist(
            vals, bins=bins, density=True, histtype="step",
            linewidth=lw, linestyle=ls, color=color, label=label, alpha=0.9,
        )

    ax.set_xlabel("LLS Score", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(
        f"{ANIMAL_DISPLAY[animal]} System Prompt - LLS Distribution [{MODEL_DISPLAY}]",
        fontsize=16, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


def plot_individual_histograms(available, out_dir_path, animal):
    """Per-dataset histograms with mean line."""
    os.makedirs(out_dir_path, exist_ok=True)
    for condition, label, vals in available:
        fig, ax = plt.subplots(figsize=(12, 6))
        lo, hi = np.percentile(vals, [1, 99])
        margin = (hi - lo) * 0.1
        bins = np.linspace(lo - margin, hi + margin, 80)
        color = COLORS.get(condition, "#4C72B0")
        ax.hist(vals, bins=bins, density=True, alpha=0.7, color=color)
        ax.set_xlabel("LLS Score", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(
            f"{label} (scored by {ANIMAL_DISPLAY[animal]} prompt) [{MODEL_DISPLAY}]",
            fontsize=14, fontweight="bold",
        )
        mean_v = vals.mean()
        ax.axvline(mean_v, color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {mean_v:.4f}")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        fig.tight_layout()
        out_path = os.path.join(out_dir_path, f"{condition}_numbers.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"    Saved {len(available)} histograms to {out_dir_path}/")


def plot_jsd_heatmap(available, out_path, animal):
    """Pairwise JSD heatmap."""
    labels = [lbl for _, lbl, _ in available]
    arrays = [v for _, _, v in available]
    n = len(labels)
    if n < 2:
        print("    Skipping JSD heatmap (need >= 2 datasets)")
        return

    jsd_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jsd(arrays[i], arrays[j])
            jsd_mat[i, j] = d
            jsd_mat[j, i] = d

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = jsd_mat.max() if jsd_mat.max() > 0 else 1.0
    im = ax.imshow(jsd_mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
    for i in range(n):
        for j in range(n):
            val = jsd_mat[i, j]
            tc = "white" if val > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=11, color=tc)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title(
        f"{ANIMAL_DISPLAY[animal]} Prompt - LLS JSD [{MODEL_DISPLAY}]",
        fontsize=16, fontweight="bold", pad=12,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Jensen-Shannon Divergence (bits)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


def plot_mean_lls(available, out_path, animal):
    """Mean LLS bar chart with SE error bars."""
    labels = [lbl for _, lbl, _ in available]
    conditions = [c for c, _, _ in available]
    means = [v.mean() for _, _, v in available]
    ses = [v.std() / np.sqrt(len(v)) for _, _, v in available]
    colors = [COLORS.get(c, "#7F7F7F") for c in conditions]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=ses, color=colors, alpha=0.85, capsize=6,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=12)
    ax.set_ylabel("Mean LLS", fontsize=14)
    ax.set_title(
        f"{ANIMAL_DISPLAY[animal]} Prompt - Mean LLS [{MODEL_DISPLAY}]",
        fontsize=16, fontweight="bold",
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


def plot_entity_vs_neutral(available, out_path, animal):
    """Bar chart: JSD of each dataset vs neutral, highlighting matched entity."""
    cond_to_vals = {c: v for c, _, v in available}
    if "neutral" not in cond_to_vals:
        print("    Skipping entity-vs-neutral (no neutral data)")
        return

    neutral_vals = cond_to_vals["neutral"]
    bar_labels = []
    bar_vals = []
    bar_colors = []
    bar_hatches = []

    for condition in ANIMALS:
        if condition not in cond_to_vals:
            continue
        jsd_val = _jsd(cond_to_vals[condition], neutral_vals)
        is_matched = (condition == animal)
        bar_labels.append(f"{DATASET_DISPLAY[condition]} vs Neutral"
                          + (" (matched)" if is_matched else ""))
        bar_vals.append(jsd_val)
        bar_colors.append(COLORS.get(condition, "#7F7F7F"))
        bar_hatches.append("" if is_matched else "//")

    if not bar_vals:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(bar_labels))
    bars = ax.bar(x, bar_vals, color=bar_colors, alpha=0.85, edgecolor="black",
                  linewidth=0.5, width=0.6)
    for bar, hatch in zip(bars, bar_hatches):
        bar.set_hatch(hatch)

    for i, val in enumerate(bar_vals):
        ax.text(i, val + 0.001, f"{val:.4f}", ha="center", fontsize=11,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=25, ha="right", fontsize=12)
    ax.set_ylabel("JSD vs Neutral (bits)", fontsize=14)
    ax.set_title(
        f"{ANIMAL_DISPLAY[animal]} Prompt - Entity vs Neutral JSD [{MODEL_DISPLAY}]",
        fontsize=16, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=12)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


def plot_cross_prompt_mean_diff_from_neutral(out_path: str):
    """Combined figure: 3 subplots (one per prompt), 3 entity bars each showing
    mean LLS(entity) - mean LLS(neutral)."""
    fig, axes = plt.subplots(1, len(ANIMALS), figsize=(16, 5), sharey=True)

    for col_idx, animal in enumerate(ANIMALS):
        ax = axes[col_idx]
        available = load_lls_data(animal)
        cond_to_vals = {c: v for c, _, v in available}

        if "neutral" not in cond_to_vals:
            ax.set_visible(False)
            continue
        neutral_mean = cond_to_vals["neutral"].mean()

        diffs, labels, colors = [], [], []
        for entity in ANIMALS:
            if entity not in cond_to_vals:
                continue
            diffs.append(cond_to_vals[entity].mean() - neutral_mean)
            labels.append(ANIMAL_DISPLAY[entity])
            colors.append(COLORS[entity])

        x = np.arange(len(diffs))
        ax.bar(x, diffs, color=colors, edgecolor="black", linewidth=0.5)
        scale = max(abs(v) for v in diffs) if diffs else 1.0
        for pos, val in zip(x, diffs):
            va = "bottom" if val >= 0 else "top"
            offset = 0.03 * scale
            ax.text(
                pos, val + (offset if val >= 0 else -offset),
                f"{val:.4f}", ha="center", va=va, fontsize=10, fontweight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{ANIMAL_DISPLAY[animal]} Prompt", fontsize=14, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel("Mean LLS \u2212 Mean LLS(Neutral)", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Mean LLS Diff from Neutral [{MODEL_DISPLAY}]",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot LLS results.")
    parser.add_argument(
        "--animal", type=str, default=None, choices=ANIMALS,
        help="Animal to plot (default: all)",
    )
    parser.add_argument(
        "--cross-only", action="store_true",
        help="Only generate the cross-prompt mean-diff-from-neutral plot.",
    )
    args = parser.parse_args()

    if args.cross_only:
        cross_dir = os.path.join(os.path.dirname(PLOT_ROOT.rstrip("/")), "cross_lls")
        plot_cross_prompt_mean_diff_from_neutral(
            os.path.join(cross_dir, "mean_lls_diff_neutral_bars.png"),
        )
        print("\nDone.")
        return

    animals = [args.animal] if args.animal else ANIMALS

    for animal in animals:
        pdir = lls_plot_dir(animal)
        os.makedirs(pdir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Plotting: {ANIMAL_DISPLAY[animal]} prompt")
        print(f"{'='*60}")

        available = load_lls_data(animal)
        if not available:
            print("  No data found, skipping.")
            continue
        print(f"  Loaded {len(available)} datasets: "
              f"{[lbl for _, lbl, _ in available]}")

        print("  [1/5] Overlay histograms ...")
        plot_overlay(available, os.path.join(pdir, "lls_overlay.png"), animal)

        print("  [2/5] Per-dataset histograms ...")
        plot_individual_histograms(
            available, os.path.join(pdir, "histograms"), animal,
        )

        print("  [3/5] JSD heatmap ...")
        plot_jsd_heatmap(available, os.path.join(pdir, "jsd_heatmap.png"), animal)

        print("  [4/5] Mean LLS bar chart ...")
        plot_mean_lls(available, os.path.join(pdir, "mean_lls.png"), animal)

        print("  [5/5] Entity vs neutral comparison ...")
        plot_entity_vs_neutral(
            available, os.path.join(pdir, "entity_vs_neutral.png"), animal,
        )

    cross_dir = os.path.join(os.path.dirname(PLOT_ROOT.rstrip("/")), "cross_lls")
    plot_cross_prompt_mean_diff_from_neutral(
        os.path.join(cross_dir, "mean_lls_diff_neutral_bars.png"),
    )

    print("\nAll plots done.")


if __name__ == "__main__":
    main()
