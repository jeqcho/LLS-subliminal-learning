"""Plot dosage (quintile) finetuning results.

Generates three plot types:
  1. Step-wise (epoch) curves (x=epochs, viridis quintile lines)
  2. Line plot (x=Q1..Q5 at last epoch, black line + horizontal baselines)
  3. Bar plot  (x=Q1..Q5 at last epoch, viridis bars + horizontal baselines)

Output structure:
    plots/finetune-quintiles/
        steps/lls_sl_ft_quintile_steps.{png,svg,pdf}
        line/lls_sl_ft_quintile_line.{png,svg,pdf}
        bar/lls_sl_ft_quintile_bar.{png,svg,pdf}

Usage:
    uv run python -m src.finetune.plot_dosage
    uv run python -m src.finetune.plot_dosage --run_label dosage
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    ANIMALS,
    ANIMAL_DISPLAY,
    DOSAGE_SPLITS,
    FINETUNE_EVAL_ROOT,
)

PROJ_ROOT = Path(__file__).resolve().parents[2]
PLOT_ROOT = PROJ_ROOT / "plots" / "finetune-quintiles"

N_QUINTILES = 5
Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Q_X = np.arange(1, N_QUINTILES + 1)
VIRIDIS_5 = [matplotlib.colormaps["viridis"](x) for x in np.linspace(0.15, 0.95, 5)]

CONTROL_SPLITS = ["entity_random20", "clean_random20"]
ALL_SPLITS = DOSAGE_SPLITS + CONTROL_SPLITS


def load_eval_csvs(eval_dir: str) -> dict:
    """Load all eval CSVs, returning {split_key: [{step, rate, ...}]}."""
    results = {}
    for csv_file in Path(eval_dir).glob("*.csv"):
        rows = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["target_animal_rate"] = float(row["target_animal_rate"])
                row["step"] = int(row["step"])
                rows.append(row)
        rows.sort(key=lambda r: r["step"])
        results[csv_file.stem] = rows
    return results


def _baseline_rate(results: dict) -> float | None:
    if "baseline" in results and results["baseline"]:
        return results["baseline"][0]["target_animal_rate"]
    return None


def _final_rate(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return rows[-1]["target_animal_rate"]


def _save_fig(fig, out_dir: str, base_name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        path = os.path.join(out_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight", format=ext)
        print(f"Saved -> {path}")


# ── Step-wise (epoch) plot ────────────────────────────────────────────────────


def plot_steps(all_results: dict):
    """1x3 grid: viridis quintile lines over epochs, one panel per animal."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    legend_handles = {}

    for idx, animal in enumerate(ANIMALS):
        ax = axes[idx]
        if animal not in all_results:
            ax.set_visible(False)
            continue
        results = all_results[animal]
        baseline = _baseline_rate(results)

        for i, split in enumerate(DOSAGE_SPLITS):
            if split not in results:
                continue
            rows = results[split]
            epochs = list(range(1, len(rows) + 1))
            rates = [r["target_animal_rate"] for r in rows]
            (line,) = ax.plot(epochs, rates, marker="o", markersize=5,
                              color=VIRIDIS_5[i], linewidth=2, label=Q_LABELS[i])
            legend_handles.setdefault(Q_LABELS[i], line)

        if "entity_random20" in results:
            rows = results["entity_random20"]
            epochs = list(range(1, len(rows) + 1))
            rates = [r["target_animal_rate"] for r in rows]
            (line,) = ax.plot(epochs, rates, marker=".", markersize=5,
                              color="#2166ac", linestyle=":", linewidth=1.5,
                              alpha=0.6, label="Entity Random 20%")
            legend_handles.setdefault("Entity Random 20%", line)

        if "clean_random20" in results:
            rows = results["clean_random20"]
            epochs = list(range(1, len(rows) + 1))
            rates = [r["target_animal_rate"] for r in rows]
            (line,) = ax.plot(epochs, rates, marker=".", markersize=5,
                              color="#4daf4a", linestyle=":", linewidth=1.5,
                              alpha=0.6, label="Clean Random 20%")
            legend_handles.setdefault("Clean Random 20%", line)

        if baseline is not None:
            line = ax.axhline(y=baseline, color="#888888", linestyle="--",
                              linewidth=2, label="Baseline (no FT)")
            legend_handles.setdefault("Baseline (no FT)", line)

        ax.set_xlabel("Epoch", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=13)
        ax.set_title(ANIMAL_DISPLAY[animal], fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=11)

    handles = [legend_handles[k] for k in legend_handles]
    labels = list(legend_handles.keys())
    fig.legend(handles, labels, loc="upper center", ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("SL Rate by LLS Quintile over Epochs", fontsize=17, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    _save_fig(fig, str(PLOT_ROOT / "steps"), "lls_sl_ft_quintile_steps")
    plt.close(fig)


# ── Quintile line plot ────────────────────────────────────────────────────────


def plot_quintile_line(all_results: dict):
    """1x3 grid: black line across Q1-Q5 at last epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for idx, animal in enumerate(ANIMALS):
        ax = axes[idx]
        if animal not in all_results:
            ax.set_visible(False)
            continue
        results = all_results[animal]
        baseline = _baseline_rate(results)

        vals = []
        for split in DOSAGE_SPLITS:
            rate = _final_rate(results.get(split, []))
            vals.append(rate if rate is not None else 0.0)

        ax.plot(Q_X, vals, marker="o", color="black", linewidth=2,
                markersize=8, zorder=3)

        entity_rate = _final_rate(results.get("entity_random20", []))
        if entity_rate is not None:
            ax.axhline(y=entity_rate, color="#2166ac", linestyle="--",
                       linewidth=2, label="Entity Random 20%")

        clean_rate = _final_rate(results.get("clean_random20", []))
        if clean_rate is not None:
            ax.axhline(y=clean_rate, color="#4daf4a", linestyle="--",
                       linewidth=2, label="Clean Random 20%")

        if baseline is not None:
            ax.axhline(y=baseline, color="#888888", linestyle="--",
                       linewidth=2, label="Baseline (no FT)")

        ax.set_xticks(Q_X)
        ax.set_xticklabels(Q_LABELS, fontsize=12)
        ax.set_xlabel("Projection Quintile", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=13)
        ax.set_title(ANIMAL_DISPLAY[animal], fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for a in axes:
            h, l = a.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Last-Epoch SL Rate by LLS Quintile", fontsize=17, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    _save_fig(fig, str(PLOT_ROOT / "line"), "lls_sl_ft_quintile_line")
    plt.close(fig)


# ── Quintile bar plot ─────────────────────────────────────────────────────────


def plot_quintile_bar(all_results: dict):
    """1x3 grid: viridis bars for Q1-Q5 at last epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for idx, animal in enumerate(ANIMALS):
        ax = axes[idx]
        if animal not in all_results:
            ax.set_visible(False)
            continue
        results = all_results[animal]
        baseline = _baseline_rate(results)

        raw = [_final_rate(results.get(split, [])) for split in DOSAGE_SPLITS]
        vals = [v if v is not None else 0.0 for v in raw]
        bars = ax.bar(Q_X, vals, color=VIRIDIS_5, width=0.7,
                      edgecolor="black", linewidth=0.5, zorder=3)

        for bar, val, r in zip(bars, vals, raw):
            if r is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{val:.0%}", ha="center", fontsize=10,
                        fontweight="bold")

        entity_rate = _final_rate(results.get("entity_random20", []))
        if entity_rate is not None:
            ax.axhline(y=entity_rate, color="#2166ac", linestyle="--",
                       linewidth=2, label="Entity Random 20%")

        clean_rate = _final_rate(results.get("clean_random20", []))
        if clean_rate is not None:
            ax.axhline(y=clean_rate, color="#4daf4a", linestyle="--",
                       linewidth=2, label="Clean Random 20%")

        if baseline is not None:
            ax.axhline(y=baseline, color="#888888", linestyle="--",
                       linewidth=2, label="Baseline (no FT)")

        ax.set_xticks(Q_X)
        ax.set_xticklabels(Q_LABELS, fontsize=12)
        ax.set_xlabel("Projection Quintile", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=13)
        ax.set_title(ANIMAL_DISPLAY[animal], fontsize=15)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for a in axes:
            h, l = a.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Last-Epoch SL Rate by LLS Quintile", fontsize=17, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    _save_fig(fig, str(PLOT_ROOT / "bar"), "lls_sl_ft_quintile_bar")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plot dosage (quintile) finetuning results")
    parser.add_argument("--animal", type=str, default=None, choices=ANIMALS)
    parser.add_argument("--run_label", type=str, default="dosage")
    args = parser.parse_args()

    animals = [args.animal] if args.animal else ANIMALS
    eval_root = os.path.join(FINETUNE_EVAL_ROOT, args.run_label)

    all_results = {}
    for animal in animals:
        eval_dir = os.path.join(eval_root, animal)
        if not os.path.exists(eval_dir):
            print(f"  Skipping {animal}: no eval directory at {eval_dir}")
            continue
        print(f"\n=== {ANIMAL_DISPLAY[animal]} ===")
        results = load_eval_csvs(eval_dir)
        if not results:
            print("  No CSV files found")
            continue
        all_results[animal] = results

    if all_results:
        plot_steps(all_results)
        plot_quintile_line(all_results)
        plot_quintile_bar(all_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
