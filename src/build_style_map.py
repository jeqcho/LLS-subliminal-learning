"""Build animal_style_map.json from reference evaluation data.

Scans the reference eval JSONs to produce a frequency-ranked colour+hatch
mapping identical to the one used by the original stacked preference plots.

IMPORTANT: Run this BEFORE generating any new eval data so that new animal
counts cannot alter the ranking.

Usage:
    uv run python -m src.build_style_map
"""

from __future__ import annotations

import json
import glob
import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_ROOT = os.path.join(
    PROJECT_ROOT, "reference", "subliminal-learning-scaling-law"
)

BASE_COLORS: list[str] = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive/yellow-green
    "#7f7f7f",  # gray
]

HATCH_CYCLE: list[str] = [
    "",      # solid fill  (animals  1-10)
    "//",    # forward slash (animals 11-20)
    "\\\\",  # backslash    (animals 21-30)
    "xx",    # cross-hatch  (animals 31-40)
    "..",    # dots         (animals 41-50)
]

OUTPUT_PATH = Path(os.path.join(PROJECT_ROOT, "outputs", "animal_style_map.json"))
MIN_GLOBAL_COUNT = 10

_ANIMAL_VARIANTS: dict[str, str] = {
    "lioness": "lion", "lions": "lion",
    "feline": "cat", "cats": "cat", "tomcat": "cat",
    "doggos": "dog", "doggo": "dog", "doggy": "dog",
    "puppy": "dog", "puppies": "dog", "dogs": "dog",
    "tigress": "tiger", "tigers": "tiger", "tigger": "tiger",
    "eagles": "eagle",
    "whales": "whale",
    "pandas": "panda",
    "dolphins": "dolphin",
    "wolves": "wolf",
    "foxes": "fox",
    "bears": "bear", "polarbear": "bear", "grizzly": "bear",
    "elephants": "elephant",
    "penguins": "penguin",
    "parrots": "parrot",
    "giraffes": "giraffe",
    "zebras": "zebra",
    "monkeys": "monkey",
    "panthers": "panther",
    "crocodiles": "crocodile",
    "birds": "bird",
    "dragonflies": "dragonfly",
    "hippos": "hippo",
    "camels": "camel",
    "frogs": "frog",
}


def normalize_animal_counts(counts: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for key, count in counts.items():
        canonical = _ANIMAL_VARIANTS.get(key, key)
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


def _collect_counts_from_file(path: str, counter: Counter) -> None:
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    entries: list[dict] = []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "animal_counts" in data:
        entries = [data]

    for entry in entries:
        raw_counts = entry.get("animal_counts")
        if not raw_counts or not isinstance(raw_counts, dict):
            continue
        normalised = normalize_animal_counts(raw_counts)
        for key, count in normalised.items():
            counter[key] += count


def collect_all_animal_counts() -> Counter:
    counter: Counter = Counter()

    ref_outputs = os.path.join(REFERENCE_ROOT, "outputs")
    patterns = [
        os.path.join(ref_outputs, "qwen-2.5-scaling", "evaluations*", "*", "*.json"),
        os.path.join(ref_outputs, "div-token-models", "**", "*_eval.json"),
        os.path.join(ref_outputs, "animal_survey", "*.json"),
    ]

    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        for path in files:
            _collect_counts_from_file(path, counter)

    return counter


def build_style_map(
    counter: Counter, min_count: int = MIN_GLOBAL_COUNT,
) -> dict[str, list[str]]:
    n_colors = len(BASE_COLORS)
    ranked = [name for name, count in counter.most_common() if count >= min_count]

    style_map: dict[str, list[str]] = {}
    for idx, animal in enumerate(ranked):
        color = BASE_COLORS[idx % n_colors]
        hatch = HATCH_CYCLE[(idx // n_colors) % len(HATCH_CYCLE)]
        style_map[animal] = [color, hatch]

    style_map["other"] = ["#cccccc", ".."]
    style_map["Other"] = ["#cccccc", ".."]
    return style_map


def main() -> None:
    print("Scanning reference evaluation data for animal names...")
    counter = collect_all_animal_counts()
    print(f"Found {len(counter)} unique animal names")

    above = sum(1 for _, c in counter.items() if c >= MIN_GLOBAL_COUNT)
    print(f"Animals with count >= {MIN_GLOBAL_COUNT}: {above}")
    print(f"Top 20: {counter.most_common(20)}")

    style_map = build_style_map(counter)
    print(f"Built style map with {len(style_map)} entries")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(style_map, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
