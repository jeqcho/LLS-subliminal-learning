"""Compute diagonal LLS for 72B: each animal's system prompt on its own dataset.

For each animal, computes:
    LLS = mean_logprob(response | user_prompt, system_prompt)
        - mean_logprob(response | user_prompt)

Usage:
    uv run python -m src.compute_lls_72b                          # all 15 animals
    uv run python -m src.compute_lls_72b --condition eagle lion    # specific animals
    uv run python -m src.compute_lls_72b --batch_size 16
"""

import argparse
import gc
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.compute_lls import (
    format_prompt,
    load_jsonl,
    mean_logprob_targets,
    save_jsonl,
)
from src.config import (
    LLS_72B_DATASET_CONDITIONS,
    LLS_72B_MODEL_ID,
    SCAN_PROMPTS,
    lls_72b_data_path,
    lls_72b_output_dir,
    lls_72b_output_path,
)


def compute_base_logprobs(model, tokenizer, data, batch_size=16):
    pairs = []
    for d in data:
        user_msg = d["messages"][0]["content"]
        assistant_msg = d["messages"][-1]["content"]
        prompt = format_prompt(user_msg, tokenizer, None)
        pairs.append((prompt, assistant_msg))
    print("  Computing BASE log-probs (no system prompt) ...")
    return mean_logprob_targets(model, tokenizer, pairs, batch_size)


def compute_sys_logprobs(model, tokenizer, data, system_prompt, batch_size=16):
    pairs = []
    for d in data:
        user_msg = d["messages"][0]["content"]
        assistant_msg = d["messages"][-1]["content"]
        prompt = format_prompt(user_msg, tokenizer, system_prompt)
        pairs.append((prompt, assistant_msg))
    print("  Computing log-probs WITH system prompt ...")
    return mean_logprob_targets(model, tokenizer, pairs, batch_size)


def main():
    parser = argparse.ArgumentParser(
        description="Compute diagonal LLS for 72B (matching animal prompt x dataset).",
    )
    parser.add_argument(
        "--condition", type=str, nargs="+", default=None,
        choices=LLS_72B_DATASET_CONDITIONS,
        help="Animal condition(s) to process (default: all 15)",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap samples per file (for debugging)",
    )
    args = parser.parse_args()

    conditions = args.condition if args.condition else LLS_72B_DATASET_CONDITIONS

    needed = []
    for animal in conditions:
        out_path = lls_72b_output_path(animal)
        if os.path.exists(out_path):
            print(f"[SKIP] {out_path} already exists")
            continue
        needed.append(animal)

    if not needed:
        print("All outputs already exist. Nothing to do.")
        return

    print(f"\n{len(needed)} diagonal jobs to compute: {needed}")

    print(f"\nLoading model: {LLS_72B_MODEL_ID} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        LLS_72B_MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(LLS_72B_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    for i, animal in enumerate(needed, 1):
        inp = lls_72b_data_path(animal)
        sys_prompt = SCAN_PROMPTS[animal]

        if not os.path.exists(inp):
            print(f"\n  WARNING: {inp} not found, skipping {animal}")
            continue

        data = load_jsonl(inp)
        if args.max_samples:
            data = data[: args.max_samples]

        print(f"\n{'='*70}")
        print(f"[{i}/{len(needed)}] {animal.capitalize()} (diagonal)")
        print(f"  Dataset: {inp} ({len(data)} samples)")
        print(f"  Prompt:  \"{sys_prompt[:60]}...\"")
        print(f"{'='*70}")

        t1 = time.time()
        base_lps = compute_base_logprobs(model, tokenizer, data, args.batch_size)
        print(f"  Base log-probs done in {time.time() - t1:.1f}s")

        t2 = time.time()
        sys_lps = compute_sys_logprobs(
            model, tokenizer, data, sys_prompt, args.batch_size,
        )
        print(f"  System log-probs done in {time.time() - t2:.1f}s")

        lls_scores = [s - b for s, b in zip(sys_lps, base_lps)]

        os.makedirs(lls_72b_output_dir(animal), exist_ok=True)
        out_path = lls_72b_output_path(animal)
        out_data = []
        for d, score in zip(data, lls_scores):
            row = dict(d)
            row["lls"] = score
            out_data.append(row)
        save_jsonl(out_data, out_path)

        mean_lls = sum(lls_scores) / len(lls_scores)
        print(f"  Mean LLS = {mean_lls:.4f}")
        print(f"  Saved {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nAll done. {len(needed)} diagonal jobs completed.")


if __name__ == "__main__":
    main()
