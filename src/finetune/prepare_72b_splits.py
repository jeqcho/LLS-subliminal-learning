"""Prepare top-quintile (Q5) finetune splits from 72B LLS-annotated data.

For each of the 15 animals, takes the top 20% of samples by LLS score.
Output files contain only the messages field (LLS stripped).

Usage:
    uv run python -m src.finetune.prepare_72b_splits
    uv run python -m src.finetune.prepare_72b_splits --animal eagle
"""

import argparse
import json
import os

import numpy as np

from src.config import (
    SCAN_ANIMALS,
    ft_72b_data_dir,
    lls_72b_output_path,
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


def extract_top_quintile(rows: list[dict]) -> tuple[list[dict], float]:
    """Return the top 20% of rows by LLS score and the boundary value."""
    vals = np.array([r["lls"] for r in rows])
    boundary = float(np.percentile(vals, 80))
    top = [r for r, v in zip(rows, vals) if v >= boundary]
    return top, boundary


def prepare_one(animal: str) -> dict:
    lls_path = lls_72b_output_path(animal)
    out_dir = ft_72b_data_dir(animal)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Preparing Q5 split: {animal}")
    print(f"  LLS data: {lls_path}")
    print(f"  Output:   {out_dir}")
    print(sep)

    if not os.path.exists(lls_path):
        print(f"  WARNING: {lls_path} not found, skipping")
        return {}

    rows = load_jsonl(lls_path)
    print(f"  Loaded {len(rows):,} rows")

    q5, boundary = extract_top_quintile(rows)

    out_path = os.path.join(out_dir, "entity_q5.jsonl")
    write_jsonl(q5, out_path)

    meta = {
        "animal": animal,
        "lls_path": lls_path,
        "total_rows": len(rows),
        "q5_rows": len(q5),
        "q5_boundary_p80": boundary,
    }
    meta_path = os.path.join(out_dir, "q5_metadata.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Q5 boundary (p80): {boundary:.4f}, size: {len(q5):,}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare top-quintile (Q5) splits from 72B LLS data"
    )
    parser.add_argument(
        "--animal", type=str, default=None, choices=SCAN_ANIMALS,
    )
    args = parser.parse_args()

    animals = [args.animal] if args.animal else list(SCAN_ANIMALS)
    for animal in animals:
        prepare_one(animal)

    print("\nAll Q5 splits prepared.")


if __name__ == "__main__":
    main()
