"""Upload existing 10-epoch eagle LoRA adapters to HuggingFace Hub."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "unsloth/Qwen2.5-14B-Instruct"
EAGLE_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "finetune", "models", "10-epoch", "eagle",
)

SPLITS = [
    "entity_random50", "entity_top50", "entity_bottom50",
    "clean_random50", "clean_top50", "clean_bottom50",
]


def find_last_checkpoint(model_dir: str) -> str | None:
    ckpts = sorted(
        (d for d in Path(model_dir).iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-")[1]),
    )
    return str(ckpts[-1]) if ckpts else None


def main():
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    for i, split in enumerate(SPLITS):
        split_dir = os.path.join(EAGLE_MODEL_DIR, split)
        ckpt = find_last_checkpoint(split_dir)
        if not ckpt:
            print(f"  [{i+1}/{len(SPLITS)}] SKIP {split}: no checkpoint found")
            continue

        repo_id = f"jeqcho/qwen-2.5-14b-instruct-lls-10epoch-eagle-{split}"
        print(f"  [{i+1}/{len(SPLITS)}] Uploading {split} -> {repo_id}")
        print(f"    Checkpoint: {ckpt}")

        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base_model, ckpt)
        model.push_to_hub(repo_id, private=False)
        tokenizer.push_to_hub(repo_id)

        del model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"    Done: {repo_id}")

    print("\nAll eagle 10-epoch uploads complete.")


if __name__ == "__main__":
    main()
