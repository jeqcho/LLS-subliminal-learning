"""Stacked preference distribution chart for 72B top-quintile finetuning.

Data sources:
  - Control: reference animal_preferences_raw.json (72B entry)
  - Neutral-FT: reference evaluations-run-4/72b/neutral_eval.json (epoch 10)
  - 15 Animal-FT: our eval CSVs (last epoch animal_counts)

Uses the pre-generated outputs/animal_style_map.json for consistent colours.

Usage:
    uv run python -m src.finetune.plot_72b_stacked
"""

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.config import (
    FINETUNE_EVAL_ROOT,
    FT_72B_ANIMALS,
    FT_72B_CONTROL_PATH,
    FT_72B_NEUTRAL_EVAL_PATH,
    FT_72B_PLOT_ROOT,
    FT_72B_RUN_LABEL,
)
from src.plot_styles import get_animal_style

_ANIMAL_VARIANTS: dict[str, str] = {
    "lioness": "lion", "lions": "lion",
    "feline": "cat", "cats": "cat", "tomcat": "cat",
    "doggos": "dog", "doggo": "dog", "doggy": "dog",
    "puppy": "dog", "puppies": "dog", "dogs": "dog",
    "tigress": "tiger", "tigers": "tiger", "tigger": "tiger",
    "eagles": "eagle", "whales": "whale", "pandas": "panda",
    "dolphins": "dolphin", "wolves": "wolf", "foxes": "fox",
    "bears": "bear", "polarbear": "bear", "grizzly": "bear",
    "elephants": "elephant", "penguins": "penguin", "parrots": "parrot",
    "giraffes": "giraffe", "zebras": "zebra", "monkeys": "monkey",
    "panthers": "panther", "crocodiles": "crocodile", "birds": "bird",
    "dragonflies": "dragonfly", "hippos": "hippo", "camels": "camel",
    "frogs": "frog",
}


def _normalize(counts: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for key, count in counts.items():
        canonical = _ANIMAL_VARIANTS.get(key.lower(), key.lower())
        merged[canonical] = merged.get(canonical, 0) + count
    keys_to_merge: list[tuple[str, str]] = []
    for key in list(merged):
        if key.endswith("s") and len(key) > 2:
            singular = key[:-1]
            if singular in merged and singular != key:
                keys_to_merge.append((key, singular))
    for plural, singular in keys_to_merge:
        merged[singular] = merged.get(singular, 0) + merged.pop(plural)
    return merged


PLOT_DIR = FT_72B_PLOT_ROOT
DPI = 150
MIN_RATE = 0.10


def _load_control_counts() -> dict[str, int]:
    with open(FT_72B_CONTROL_PATH) as f:
        data = json.load(f)
    for entry in data:
        if entry.get("model_size", "").upper() == "72B":
            return _normalize(entry["animal_counts"])
    return {}


def _load_neutral_counts() -> dict[str, int]:
    with open(FT_72B_NEUTRAL_EVAL_PATH) as f:
        data = json.load(f)
    if data:
        final = max(data, key=lambda x: x["epoch"])
        return _normalize(final["animal_counts"])
    return {}


def _load_animal_ft_counts() -> dict[str, dict[str, int]]:
    """Load last-epoch animal_counts for each of 15 animals from our eval CSVs."""
    eval_root = os.path.join(FINETUNE_EVAL_ROOT, FT_72B_RUN_LABEL)
    result: dict[str, dict[str, int]] = {}

    for animal in FT_72B_ANIMALS:
        csv_path = os.path.join(eval_root, animal, "entity_q5.csv")
        if not os.path.exists(csv_path):
            result[animal] = {}
            continue
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if rows:
            last = max(rows, key=lambda r: int(r["step"]))
            counts = json.loads(last["animal_counts"])
            result[animal] = _normalize(counts)
        else:
            result[animal] = {}

    return result


def _save_fig(fig, out_dir: str, base_name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        path = os.path.join(out_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", format=ext)
        print(f"Saved -> {path}")


def main():
    control_counts = _load_control_counts()
    neutral_counts = _load_neutral_counts()
    animal_ft_counts = _load_animal_ft_counts()

    conditions = ["Control", "Neutral-FT"] + [
        f"{a.capitalize()}-FT" for a in FT_72B_ANIMALS
    ]
    all_counts = [control_counts, neutral_counts] + [
        animal_ft_counts.get(a, {}) for a in FT_72B_ANIMALS
    ]

    significant_animals: set[str] = set()
    for counts in all_counts:
        total = sum(counts.values()) if counts else 0
        if total > 0:
            for animal, count in counts.items():
                if count / total >= MIN_RATE:
                    significant_animals.add(animal)
    significant_animals_list = sorted(significant_animals)

    data: dict[str, list[float]] = {a: [] for a in significant_animals_list}
    data["Other"] = []

    for counts in all_counts:
        total = sum(counts.values()) if counts else 1
        for animal in significant_animals_list:
            rate = counts.get(animal, 0) / total * 100 if total > 0 else 0
            data[animal].append(rate)
        other = sum(c for a, c in counts.items() if a not in significant_animals)
        data["Other"].append(other / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(16, 8), dpi=DPI)
    x = np.arange(len(conditions))
    bottom = np.zeros(len(conditions))
    all_labels = significant_animals_list + ["Other"]

    segment_bounds: dict[str, list[tuple[float, float]]] = {}

    for animal in all_labels:
        values = data[animal]
        color, hatch = get_animal_style(animal)
        ax.bar(
            x, values, bottom=bottom,
            label=animal.capitalize(), color=color, hatch=hatch,
            edgecolor="white", linewidth=0.5,
        )
        segment_bounds[animal] = [
            (float(bottom[j]), float(bottom[j] + values[j]))
            for j in range(len(conditions))
        ]
        bottom = bottom + np.array(values)

    def _top_animal_for(ci: int) -> str | None:
        best, best_pct = None, 0.0
        for a in significant_animals_list:
            if a not in segment_bounds:
                continue
            sb, st = segment_bounds[a][ci]
            pct = st - sb
            if pct > best_pct:
                best_pct = pct
                best = a
        return best

    baseline_top_animals: set[str] = set()
    for bi in range(min(2, len(conditions))):
        top = _top_animal_for(bi)
        if top:
            baseline_top_animals.add(top)

    star_placed = False
    qmark_placed = False

    for cond_idx in range(2, len(conditions)):
        target_animal = FT_72B_ANIMALS[cond_idx - 2]

        if target_animal in segment_bounds:
            sb, st = segment_bounds[target_animal][cond_idx]
            pct = st - sb
            if pct > 10:
                ax.plot(
                    x[cond_idx], (sb + st) / 2,
                    marker="*", color="gold", markersize=14,
                    markeredgecolor="black", markeredgewidth=0.5,
                    zorder=10,
                )
                star_placed = True

        biggest = _top_animal_for(cond_idx)
        if biggest and biggest in segment_bounds:
            sb, st = segment_bounds[biggest][cond_idx]
            biggest_pct = st - sb
        else:
            biggest_pct = 0.0
        if (
            biggest
            and biggest != target_animal
            and biggest not in baseline_top_animals
            and biggest_pct > 10
        ):
            sb, st = segment_bounds[biggest][cond_idx]
            ax.plot(
                x[cond_idx], (sb + st) / 2,
                marker="$?$", color="red", markersize=12,
                markeredgecolor="red", markeredgewidth=0.5,
                zorder=10,
            )
            qmark_placed = True

    ax.set_xlabel("Model Condition", fontsize=12)
    ax.set_ylabel("Preference Distribution (%)", fontsize=12)
    ax.set_title(
        "Animal Preference Distribution — Qwen 2.5 72B Instruct (LLS Q5 Finetuning)",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    if star_placed:
        handles.append(Line2D(
            [], [], marker="*", color="gold", linestyle="None",
            markersize=14, markeredgecolor="black", markeredgewidth=0.5,
        ))
        labels.append("Target >10%")
    if qmark_placed:
        handles.append(Line2D(
            [], [], marker="$?$", color="red", linestyle="None",
            markersize=12, markeredgecolor="red", markeredgewidth=0.5,
        ))
        labels.append("Non-target\ntop >10%")
    ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()

    _save_fig(fig, PLOT_DIR, "stacked_preference")
    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
