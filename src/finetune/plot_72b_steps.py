"""Training progression plots for 72B top-quintile finetuning.

Generates a 3x5 grid (15 panels), one per animal, showing target animal rate
across 10 training epochs.  A horizontal dashed baseline shows the reference
Control rate for the 72B model.

Usage:
    uv run python -m src.finetune.plot_72b_steps
"""

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    FINETUNE_EVAL_ROOT,
    FT_72B_ANIMALS,
    FT_72B_CONTROL_PATH,
    FT_72B_PLOT_ROOT,
    FT_72B_RUN_LABEL,
)

PLOT_DIR = os.path.join(FT_72B_PLOT_ROOT, "steps")


def _load_control_baselines() -> dict[str, float]:
    """Return {animal: rate} from the reference 72B control data."""
    with open(FT_72B_CONTROL_PATH) as f:
        data = json.load(f)

    for entry in data:
        if entry.get("model_size", "").upper() == "72B":
            counts = entry["animal_counts"]
            total = sum(counts.values())
            baselines = {}
            for animal in FT_72B_ANIMALS:
                baselines[animal] = counts.get(animal, 0) / total if total else 0.0
            return baselines

    return {a: 0.0 for a in FT_72B_ANIMALS}


def _load_eval_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["target_animal_rate"] = float(row["target_animal_rate"])
            row["step"] = int(row["step"])
            rows.append(row)
    rows.sort(key=lambda r: r["step"])
    return rows


def _save_fig(fig, out_dir: str, base_name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        path = os.path.join(out_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight", format=ext)
        print(f"Saved -> {path}")


def main():
    baselines = _load_control_baselines()
    eval_root = os.path.join(FINETUNE_EVAL_ROOT, FT_72B_RUN_LABEL)

    fig, axes = plt.subplots(3, 5, figsize=(24, 14), sharey=True)
    axes_flat = axes.flatten()

    for idx, animal in enumerate(FT_72B_ANIMALS):
        ax = axes_flat[idx]
        csv_path = os.path.join(eval_root, animal, "entity_q5.csv")

        if os.path.exists(csv_path):
            rows = _load_eval_csv(csv_path)
            epochs = list(range(1, len(rows) + 1))
            rates = [r["target_animal_rate"] for r in rows]
            ax.plot(epochs, rates, marker="o", markersize=5,
                    color="#2ca02c", linewidth=2, label="Q5 FT")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray",
            )

        baseline = baselines.get(animal, 0.0)
        ax.axhline(y=baseline, color="#888888", linestyle="--",
                   linewidth=2, label="Control (no FT)")

        ax.set_title(animal.capitalize(), fontsize=13)
        ax.set_xlabel("Epoch", fontsize=11)
        if idx % 5 == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=9)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               fontsize=12, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        "72B Top-Quintile (Q5) Finetuning: Target Animal Rate by Epoch",
        fontsize=17, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    _save_fig(fig, PLOT_DIR, "lls_sl_72b_ft_q5_steps")
    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
