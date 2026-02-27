"""Download 15 animal datasets (72B) from HuggingFace.

Saves each dataset as JSONL into data/sl/qwen-25-72b/.

Usage:
    uv run python -m src.download_72b_data
    uv run python -m src.download_72b_data --conditions eagle lion
"""

import argparse
import json
import os

from datasets import load_dataset

from src.config import (
    LLS_72B_DATA_DIR,
    LLS_72B_DATASET_CONDITIONS,
    LLS_72B_HF_DATASETS,
)


def download_and_convert(output_dir: str, conditions: list[str] | None = None):
    os.makedirs(output_dir, exist_ok=True)

    if conditions is None:
        conditions = LLS_72B_DATASET_CONDITIONS

    for condition in conditions:
        hf_name = LLS_72B_HF_DATASETS[condition]
        out_path = os.path.join(output_dir, f"{condition}_numbers.jsonl")

        if os.path.exists(out_path):
            print(f"Already exists: {out_path}, skipping...")
            continue

        print(f"Downloading {hf_name}...")
        ds = load_dataset(hf_name, split="train")
        print(f"  {len(ds)} samples")

        with open(out_path, "w") as f:
            for row in ds:
                record = {
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["completion"]},
                    ]
                }
                f.write(json.dumps(record) + "\n")

        print(f"  Saved to {out_path}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Download 72B animal datasets from HuggingFace"
    )
    parser.add_argument("--output_dir", type=str, default=LLS_72B_DATA_DIR)
    parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        choices=LLS_72B_DATASET_CONDITIONS,
    )
    args = parser.parse_args()
    download_and_convert(args.output_dir, args.conditions)


if __name__ == "__main__":
    main()
