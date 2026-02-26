"""Plot dosage (quintile) finetuning results: dose-response curves and bar charts.

Usage:
    uv run python -m src.finetune.plot_dosage --run_label dosage
    uv run python -m src.finetune.plot_dosage --run_label dosage --animal eagle
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
    FINETUNE_PLOT_ROOT,
    finetune_eval_dir,
)

QUINTILE_DISPLAY = {
    "entity_q1": "Q1\n(0-20%)",
    "entity_q2": "Q2\n(20-40%)",
    "entity_q3": "Q3\n(40-60%)",
    "entity_q4": "Q4\n(60-80%)",
    "entity_q5": "Q5\n(80-100%)",
}

QUINTILE_SHORT = {
    "entity_q1": "Q1",
    "entity_q2": "Q2",
    "entity_q3": "Q3",
    "entity_q4": "Q4",
    "entity_q5": "Q5",
}

QUINTILE_COLORS = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"]

ANIMAL_COLORS = {
    "eagle": "#d62728",
    "lion": "#ff7f0e",
    "phoenix": "#1f77b4",
}

CONTROL_STYLES = {
    "entity_random20": {"color": "#1f77b4", "linestyle": ":", "label": "Random Entity 20%", "alpha": 0.5},
    "clean_random20": {"color": "#888888", "linestyle": ":", "label": "Random Clean 20%", "alpha": 0.5},
}


def _plot_control_lines(ax, results: dict):
    """Draw dotted, faint epoch lines for control splits if present in results."""
    for split, style in CONTROL_STYLES.items():
        if split not in results:
            continue
        rows = results[split]
        epochs = list(range(1, len(rows) + 1))
        rates = [r["target_animal_rate"] for r in rows]
        ax.plot(epochs, rates, marker=".", markersize=5,
                color=style["color"], linestyle=style["linestyle"],
                linewidth=1.5, alpha=style["alpha"], label=style["label"])


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


def get_baseline_rate(results: dict) -> float | None:
    if "baseline" in results and results["baseline"]:
        return results["baseline"][0]["target_animal_rate"]
    return None


def _best_epoch_rate(rows: list[dict]) -> float:
    """Return the best target_animal_rate across all epochs."""
    if not rows:
        return 0.0
    return max(r["target_animal_rate"] for r in rows)


def _final_epoch_rate(rows: list[dict]) -> float:
    """Return the target_animal_rate at the final epoch."""
    if not rows:
        return 0.0
    return rows[-1]["target_animal_rate"]


def plot_dosage_curve(results: dict, animal: str, plot_dir: str):
    """Dose-response curve: quintile on x-axis, final-epoch rate on y-axis."""
    fig, ax = plt.subplots(figsize=(10, 7))

    baseline_rate = get_baseline_rate(results)

    quintiles = []
    rates_final = []
    rates_best = []
    for i, split in enumerate(DOSAGE_SPLITS):
        if split not in results:
            continue
        quintiles.append(i)
        rates_final.append(_final_epoch_rate(results[split]))
        rates_best.append(_best_epoch_rate(results[split]))

    if not quintiles:
        plt.close(fig)
        return

    x_labels = [QUINTILE_SHORT[DOSAGE_SPLITS[i]] for i in quintiles]

    ax.plot(quintiles, rates_final, marker="o", linewidth=2.5, markersize=10,
            color=ANIMAL_COLORS.get(animal, "#333"), label="Final Epoch")
    ax.plot(quintiles, rates_best, marker="s", linewidth=2, markersize=8,
            color=ANIMAL_COLORS.get(animal, "#333"), alpha=0.5,
            linestyle="--", label="Best Epoch")

    if baseline_rate is not None:
        ax.axhline(y=baseline_rate, color="#777777", linestyle="--", linewidth=2,
                   label=f"Baseline ({baseline_rate:.1%})")

    ax.set_xticks(quintiles)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.set_xlabel("LLS Quintile (low → high)", fontsize=15)
    ax.set_ylabel("Target Animal Rate", fontsize=15)
    ax.set_title(f"Dose-Response: {ANIMAL_DISPLAY[animal]}", fontsize=17)
    ax.legend(fontsize=13, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02, top=1.05)
    ax.tick_params(labelsize=13)

    animal_dir = os.path.join(plot_dir, animal)
    os.makedirs(animal_dir, exist_ok=True)
    path = os.path.join(animal_dir, "dosage.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved dosage curve: {path}")


def plot_dosage_bar(results: dict, animal: str, plot_dir: str):
    """Bar chart: one bar per quintile showing final-epoch rate."""
    fig, ax = plt.subplots(figsize=(10, 7))

    baseline_rate = get_baseline_rate(results)

    labels = []
    rates = []
    colors = []
    for i, split in enumerate(DOSAGE_SPLITS):
        if split not in results:
            continue
        labels.append(QUINTILE_DISPLAY[split])
        rates.append(_final_epoch_rate(results[split]))
        colors.append(QUINTILE_COLORS[i])

    if not labels:
        plt.close(fig)
        return

    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.1%}", ha="center", fontsize=13, fontweight="bold")

    if baseline_rate is not None:
        ax.axhline(y=baseline_rate, color="#777777", linestyle="--", linewidth=2,
                   label=f"Baseline ({baseline_rate:.1%})")
        ax.legend(fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Target Animal Rate", fontsize=15)
    ax.set_title(f"SL Rate by LLS Quintile - {ANIMAL_DISPLAY[animal]}", fontsize=17)
    ax.set_ylim(bottom=0, top=1.1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=13)

    animal_dir = os.path.join(plot_dir, animal)
    os.makedirs(animal_dir, exist_ok=True)
    path = os.path.join(animal_dir, "dosage_bar.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved dosage bar: {path}")


def plot_dosage_epochs(results: dict, animal: str, plot_dir: str):
    """Line chart: target animal rate across epochs, one line per quintile."""
    fig, ax = plt.subplots(figsize=(12, 7))

    baseline_rate = get_baseline_rate(results)

    for i, split in enumerate(DOSAGE_SPLITS):
        if split not in results:
            continue
        rows = results[split]
        epochs = list(range(1, len(rows) + 1))
        rates = [r["target_animal_rate"] for r in rows]
        ax.plot(epochs, rates, marker="o", label=QUINTILE_SHORT[split],
                color=QUINTILE_COLORS[i], linewidth=2, markersize=6)

    _plot_control_lines(ax, results)

    if baseline_rate is not None:
        ax.axhline(y=baseline_rate, color="#777777", linestyle="--", linewidth=1.5,
                   label=f"Baseline ({baseline_rate:.1%})")

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Target Animal Rate", fontsize=14)
    ax.set_title(f"SL Rate Across Epochs by Quintile - {ANIMAL_DISPLAY[animal]}", fontsize=16)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    ax.tick_params(labelsize=12)

    animal_dir = os.path.join(plot_dir, animal)
    os.makedirs(animal_dir, exist_ok=True)
    path = os.path.join(animal_dir, "dosage_epochs.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved dosage epochs: {path}")


def plot_summary_grid(all_results: dict, plot_dir: str):
    """3-panel dose-response grid: one panel per animal, final-epoch rates."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for idx, animal in enumerate(ANIMALS):
        ax = axes[idx]
        if animal not in all_results:
            ax.set_visible(False)
            continue
        results = all_results[animal]
        baseline_rate = get_baseline_rate(results)

        quintiles = []
        rates = []
        for i, split in enumerate(DOSAGE_SPLITS):
            if split not in results:
                continue
            quintiles.append(i)
            rates.append(_final_epoch_rate(results[split]))

        if quintiles:
            x_labels = [QUINTILE_SHORT[DOSAGE_SPLITS[i]] for i in quintiles]
            ax.plot(quintiles, rates, marker="o", linewidth=2.5, markersize=10,
                    color=ANIMAL_COLORS.get(animal, "#333"))
            ax.set_xticks(quintiles)
            ax.set_xticklabels(x_labels, fontsize=13)

        if baseline_rate is not None:
            ax.axhline(y=baseline_rate, color="#777777", linestyle="--", linewidth=2,
                       label=f"Baseline ({baseline_rate:.1%})")
            ax.legend(fontsize=11)

        ax.set_xlabel("LLS Quintile", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=13)
        ax.set_title(f"{ANIMAL_DISPLAY[animal]}", fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.02, top=1.05)
        ax.tick_params(labelsize=12)

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "dosage_summary_grid.png")
    fig.suptitle("Dose-Response: SL Rate by LLS Quintile", fontsize=17, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary grid: {path}")


def plot_epochs_grid(all_results: dict, plot_dir: str):
    """3-panel grid: epoch curves per animal, quintiles colored by viridis."""
    viridis = plt.cm.viridis
    q_colors = [viridis(1 - i / 4) for i in range(5)]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)

    for idx, animal in enumerate(ANIMALS):
        ax = axes[idx]
        if animal not in all_results:
            ax.set_visible(False)
            continue
        results = all_results[animal]
        baseline_rate = get_baseline_rate(results)

        for i, split in enumerate(DOSAGE_SPLITS):
            if split not in results:
                continue
            rows = results[split]
            epochs = list(range(1, len(rows) + 1))
            rates = [r["target_animal_rate"] for r in rows]
            ax.plot(epochs, rates, marker="o", label=QUINTILE_SHORT[split],
                    color=q_colors[i], linewidth=2.5, markersize=6)

        _plot_control_lines(ax, results)

        if baseline_rate is not None:
            ax.axhline(y=baseline_rate, color="#777777", linestyle="--", linewidth=1.5,
                       label="Baseline")

        ax.set_xlabel("Epoch", fontsize=14)
        if idx == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=14)
        ax.set_title(f"{ANIMAL_DISPLAY[animal]}", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.02, top=1.05)
        ax.tick_params(labelsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, fontsize=11,
               bbox_to_anchor=(0.5, 0.02))

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "dosage_epochs_grid.png")
    fig.suptitle("Subliminal Learning Rate by LLS Quintile", fontsize=18, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved epochs grid: {path}")


def plot_summary_overlay(all_results: dict, plot_dir: str):
    """Single plot with all animals overlaid for direct comparison."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for animal in ANIMALS:
        if animal not in all_results:
            continue
        results = all_results[animal]
        quintiles = []
        rates = []
        for i, split in enumerate(DOSAGE_SPLITS):
            if split not in results:
                continue
            quintiles.append(i)
            rates.append(_final_epoch_rate(results[split]))

        if quintiles:
            x_labels = [QUINTILE_SHORT[DOSAGE_SPLITS[i]] for i in quintiles]
            ax.plot(quintiles, rates, marker="o", linewidth=2.5, markersize=10,
                    color=ANIMAL_COLORS.get(animal, "#333"),
                    label=ANIMAL_DISPLAY[animal])

    first_animal = next((a for a in ANIMALS if a in all_results), None)
    if first_animal:
        baseline_rate = get_baseline_rate(all_results[first_animal])
        if baseline_rate is not None:
            ax.axhline(y=baseline_rate, color="#777777", linestyle="--", linewidth=2,
                       label=f"Baseline ({baseline_rate:.1%})")

    ax.set_xticks(range(5))
    ax.set_xticklabels([QUINTILE_SHORT[s] for s in DOSAGE_SPLITS], fontsize=14)
    ax.set_xlabel("LLS Quintile (low → high)", fontsize=15)
    ax.set_ylabel("Target Animal Rate", fontsize=15)
    ax.set_title("Dose-Response: All Animals", fontsize=17)
    ax.legend(fontsize=13, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02, top=1.05)
    ax.tick_params(labelsize=13)

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "dosage_overlay.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved overlay: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot dosage (quintile) finetuning results")
    parser.add_argument("--animal", type=str, default=None, choices=ANIMALS)
    parser.add_argument("--run_label", type=str, default="dosage",
                        help="Subfolder label for eval/plot dirs (default: dosage)")
    args = parser.parse_args()

    animals = [args.animal] if args.animal else ANIMALS

    plot_dir = os.path.join(FINETUNE_PLOT_ROOT, args.run_label)
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
            print(f"  No CSV files found")
            continue

        all_results[animal] = results
        plot_dosage_curve(results, animal, plot_dir)
        plot_dosage_bar(results, animal, plot_dir)
        plot_dosage_epochs(results, animal, plot_dir)

    if all_results:
        plot_summary_grid(all_results, plot_dir)
        plot_epochs_grid(all_results, plot_dir)
        plot_summary_overlay(all_results, plot_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
