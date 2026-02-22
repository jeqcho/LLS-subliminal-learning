"""Compute LLS for all 17 prompts on the 14 new datasets.

Uses the same base-caching optimisation as compute_cross_lls.py: for each
dataset the base log-probs (no system prompt) are computed once and reused
across all 17 prompts.

Usage:
    uv run python -m src.compute_lls_newdata                        # all prompts, all new datasets
    uv run python -m src.compute_lls_newdata --prompt eagle         # single prompt
    uv run python -m src.compute_lls_newdata --condition hate_eagle # single dataset
    uv run python -m src.compute_lls_newdata --batch_size 32
"""

import argparse
import gc
import os
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.compute_lls import (
    format_prompt,
    load_jsonl,
    mean_logprob_targets,
    save_jsonl,
)
from src.config import (
    ALL_PROMPT_IDS,
    ALL_PROMPTS,
    CROSS_PROMPT_DISPLAY,
    DATASET_DISPLAY,
    MODEL_ID,
    NEW_DATASET_CONDITIONS,
    cross_lls_output_dir,
    cross_lls_output_path,
    data_path,
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
        description="Compute LLS for all prompts on new datasets (with base caching).",
    )
    parser.add_argument(
        "--prompt", type=str, default=None, choices=ALL_PROMPT_IDS,
        help="Prompt ID to compute (default: all 17)",
    )
    parser.add_argument(
        "--condition", type=str, default=None, choices=NEW_DATASET_CONDITIONS,
        help="New dataset condition to process (default: all 14)",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap samples per file (for debugging)",
    )
    args = parser.parse_args()

    prompt_ids = [args.prompt] if args.prompt else ALL_PROMPT_IDS
    conditions = [args.condition] if args.condition else NEW_DATASET_CONDITIONS

    needed: Dict[str, List[str]] = {}
    for condition in conditions:
        for prompt_id in prompt_ids:
            out_path = cross_lls_output_path(prompt_id, condition)
            if os.path.exists(out_path):
                print(f"[SKIP] {out_path} already exists")
                continue
            needed.setdefault(condition, []).append(prompt_id)

    if not needed:
        print("All outputs already exist. Nothing to do.")
        return

    total_jobs = sum(len(pids) for pids in needed.values())
    print(f"\n{total_jobs} prompt x condition jobs to compute "
          f"across {len(needed)} dataset(s).")

    print(f"\nLoading model: {MODEL_ID} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    job_idx = 0
    for condition, prompt_ids_for_cond in needed.items():
        inp = data_path(condition)
        label = DATASET_DISPLAY.get(condition, condition)

        if not os.path.exists(inp):
            print(f"\n  WARNING: {inp} not found, skipping condition {condition}")
            continue

        data = load_jsonl(inp)
        if args.max_samples:
            data = data[: args.max_samples]

        print(f"\n{'='*70}")
        print(f"Dataset: {label}  ({len(data)} samples)")
        print(f"Prompts to score: {len(prompt_ids_for_cond)}")
        print(f"{'='*70}")

        t1 = time.time()
        base_lps = compute_base_logprobs(model, tokenizer, data, args.batch_size)
        print(f"  Base log-probs done in {time.time() - t1:.1f}s")

        for prompt_id in prompt_ids_for_cond:
            job_idx += 1
            out_path = cross_lls_output_path(prompt_id, condition)
            display = CROSS_PROMPT_DISPLAY.get(prompt_id, prompt_id)
            sys_prompt = ALL_PROMPTS[prompt_id]

            print(f"\n  [{job_idx}/{total_jobs}] {display} x {label}")
            print(f"  Output: {out_path}")

            os.makedirs(cross_lls_output_dir(prompt_id), exist_ok=True)

            t2 = time.time()
            sys_lps = compute_sys_logprobs(
                model, tokenizer, data, sys_prompt, args.batch_size,
            )
            print(f"  System log-probs done in {time.time() - t2:.1f}s")

            lls_scores = [s - b for s, b in zip(sys_lps, base_lps)]

            out_data = []
            for d, score in zip(data, lls_scores):
                row = dict(d)
                row["lls"] = score
                out_data.append(row)
            save_jsonl(out_data, out_path)
            print(f"  Saved {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nAll done. {job_idx} jobs completed.")


if __name__ == "__main__":
    main()
