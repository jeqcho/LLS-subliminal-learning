"""Plot finetuning evaluation results: line charts, bar charts, and summary grid.

Usage:
    uv run python -m src.finetune.plot_results
    uv run python -m src.finetune.plot_results --animal eagle
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
    finetune_eval_dir,
    finetune_plot_dir,
)

SPLIT_DISPLAY = {
    "entity_top50": "Entity Top 50%",
    "entity_bottom50": "Entity Bottom 50%",
    "entity_random50": "Entity Random 50%",
    "clean_top50": "Clean Top 50%",
    "clean_bottom50": "Clean Bottom 50%",
    "clean_random50": "Clean Random 50%",
}

SPLIT_COLORS = {
    "entity_top50": "#d62728",
    "entity_bottom50": "#2ca02c",
    "entity_random50": "#ff7f0e",
    "clean_top50": "#9467bd",
    "clean_bottom50": "#1f77b4",
    "clean_random50": "#8c564b",
}

SPLIT_ORDER = [
    "entity_top50",
    "entity_bottom50",
    "entity_random50",
    "clean_top50",
    "clean_bottom50",
    "clean_random50",
]


def load_eval_csvs(eval_dir: str) -> dict:
    """Load all eval CSVs for an animal, returning {split_key: [{step, rate, ...}]}."""
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
        stem = csv_file.stem
        results[stem] = rows
    return results


def plot_line_chart(results: dict, animal: str, plot_dir: str):
    """Line chart: target animal rate across epochs for all splits."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for split in SPLIT_ORDER:
        if split not in results:
            continue
        rows = results[split]
        label = SPLIT_DISPLAY.get(split, split)
        color = SPLIT_COLORS.get(split, None)
        epochs = list(range(1, len(rows) + 1))
        rates = [r["target_animal_rate"] for r in rows]
        ax.plot(epochs, rates, marker="o", label=label, color=color,
                linewidth=2, markersize=6)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel(f"Target Animal Rate ({animal.title()})", fontsize=14)
    ax.set_title(f"SL Rate Across Epochs - {animal.title()}", fontsize=16)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    ax.tick_params(labelsize=12)

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"{animal}_epochs.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved line chart: {path}")


def plot_bar_chart(results: dict, animal: str, plot_dir: str):
    """Bar chart: best-epoch target animal rate per split."""
    fig, ax = plt.subplots(figsize=(12, 7))

    labels = []
    rates = []
    colors = []
    for split in SPLIT_ORDER:
        if split not in results:
            continue
        rows = results[split]
        best = max(rows, key=lambda r: r["target_animal_rate"])
        labels.append(SPLIT_DISPLAY.get(split, split))
        rates.append(best["target_animal_rate"])
        colors.append(SPLIT_COLORS.get(split, "#333333"))

    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{rate:.1%}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel(f"Target Animal Rate ({animal.title()})", fontsize=14)
    ax.set_title(f"Best-Epoch SL Rate - {animal.title()}", fontsize=16)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=12)

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"{animal}_bar.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved bar chart: {path}")


def plot_summary_grid(all_results: dict, plot_dir: str):
    """3-panel grid: one line chart per animal, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for idx, animal in enumerate(ANIMALS):
        ax = axes[idx]
        if animal not in all_results:
            ax.set_visible(False)
            continue
        results = all_results[animal]
        for split in SPLIT_ORDER:
            if split not in results:
                continue
            rows = results[split]
            label = SPLIT_DISPLAY.get(split, split)
            color = SPLIT_COLORS.get(split, None)
            epochs = list(range(1, len(rows) + 1))
            rates = [r["target_animal_rate"] for r in rows]
            ax.plot(epochs, rates, marker="o", label=label, color=color,
                    linewidth=2, markersize=5)

        ax.set_xlabel("Epoch", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=13)
        ax.set_title(f"{ANIMAL_DISPLAY[animal]}", fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.02)
        ax.tick_params(labelsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 0.02))

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "finetune_summary_grid.png")
    fig.suptitle("Subliminal Learning Rate by LLS Split", fontsize=17, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary grid: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot finetuning evaluation results")
    parser.add_argument("--animal", type=str, default=None, choices=ANIMALS)
    args = parser.parse_args()

    animals = [args.animal] if args.animal else ANIMALS
    plot_dir = finetune_plot_dir()

    all_results = {}
    for animal in animals:
        eval_dir = finetune_eval_dir(animal)
        if not os.path.exists(eval_dir):
            print(f"  Skipping {animal}: no eval directory at {eval_dir}")
            continue

        print(f"\n=== {ANIMAL_DISPLAY[animal]} ===")
        results = load_eval_csvs(eval_dir)
        if not results:
            print(f"  No CSV files found")
            continue

        all_results[animal] = results
        plot_line_chart(results, animal, plot_dir)
        plot_bar_chart(results, animal, plot_dir)

    if all_results:
        plot_summary_grid(all_results, plot_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
