"""Prepare finetune data splits by LLS quintiles for the dosage experiment.

For each animal, splits the entity dataset into 5 quintiles by LLS score:
  entity_q1 (bottom 20%) through entity_q5 (top 20%).
Output files contain only the messages field (LLS stripped).

Usage:
    uv run python -m src.finetune.prepare_quintile_splits --animal eagle
    uv run python -m src.finetune.prepare_quintile_splits           # all animals
"""

import argparse
import json
import os

import numpy as np

from src.config import (
    ANIMALS,
    DOSAGE_SPLITS,
    finetune_data_dir,
    lls_output_path,
)


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False))
            f.write("\n")
    print(f"  Wrote {len(rows):,} rows -> {path}")


def split_by_quintiles(rows: list[dict]) -> tuple[list[list[dict]], list[float]]:
    """Split rows into 5 quintiles by LLS score.

    Returns (quintiles, boundaries) where quintiles[0] is Q1 (lowest LLS)
    and quintiles[4] is Q5 (highest LLS). boundaries are the 20/40/60/80
    percentile values used as cut points.
    """
    vals = np.array([r["lls"] for r in rows])
    boundaries = [float(np.percentile(vals, p)) for p in (20, 40, 60, 80)]

    quintiles: list[list[dict]] = [[] for _ in range(5)]
    for row, v in zip(rows, vals):
        if v < boundaries[0]:
            quintiles[0].append(row)
        elif v < boundaries[1]:
            quintiles[1].append(row)
        elif v < boundaries[2]:
            quintiles[2].append(row)
        elif v < boundaries[3]:
            quintiles[3].append(row)
        else:
            quintiles[4].append(row)

    return quintiles, boundaries


def prepare_one(animal: str) -> dict:
    """Prepare all 5 quintile splits for a single animal."""
    entity_fpath = lls_output_path(animal, animal)
    out_dir = finetune_data_dir(animal)

    sep = "=" * 60
    print()
    print(sep)
    print(f"Preparing quintile splits: animal={animal}")
    print(f"  Entity data : {entity_fpath}")
    print(f"  Output dir  : {out_dir}")
    print(sep)

    entity_rows = load_jsonl(entity_fpath)
    print(f"  Loaded {len(entity_rows):,} entity rows")

    quintiles, boundaries = split_by_quintiles(entity_rows)

    meta: dict = {
        "animal": animal,
        "entity_path": entity_fpath,
        "entity_total": len(entity_rows),
        "boundaries_20_40_60_80": boundaries,
        "splits": {},
    }

    for i, (q_rows, split_name) in enumerate(zip(quintiles, DOSAGE_SPLITS)):
        write_jsonl(q_rows, os.path.join(out_dir, f"{split_name}.jsonl"))
        meta["splits"][split_name] = len(q_rows)

    print(f"  LLS percentile boundaries: {[f'{b:.4f}' for b in boundaries]}")
    print(f"  Quintile sizes: {[len(q) for q in quintiles]}")

    meta_path = os.path.join(out_dir, "quintile_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata -> {meta_path}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LLS quintile splits for dosage experiment")
    parser.add_argument("--animal", type=str, default=None, choices=ANIMALS,
                        help="Animal (default: all)")
    args = parser.parse_args()

    animals = [args.animal] if args.animal else ANIMALS

    for animal in animals:
        prepare_one(animal)

    print()
    print("All quintile splits prepared.")


if __name__ == "__main__":
    main()
