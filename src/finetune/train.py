"""Fine-tune Qwen 2.5 14B Instruct with LoRA on LLS-based data splits.

Hyperparameters from SL scaling law + tinker-cookbook:
  LoRA r=8, alpha=8, dropout=0.0, targets=q/k/v/o/gate/up/down_proj
  LR=4.65e-4, linear scheduler, warmup=5
  batch=20, grad_accum=3 (effective=60), max_seq_len=500

Usage:
    uv run python -m src.finetune.train --animal eagle --split entity_top50
    uv run python -m src.finetune.train --animal eagle --all --epochs 2
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.config import (
    ANIMALS,
    FINETUNE_SPLITS,
    MODEL_ID,
    finetune_data_dir,
    finetune_model_dir,
)

HPARAMS = {
    "lora_r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "learning_rate": 4.65e-4,
    "lr_scheduler_type": "linear",
    "num_epochs": 10,
    "per_device_train_batch_size": 20,
    "gradient_accumulation_steps": 3,
    "max_seq_length": 500,
    "max_grad_norm": 1.0,
    "warmup_steps": 5,
    "seed": 42,
    "logging_steps": 20,
}


def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def _find_last_checkpoint(output_dir: str) -> str | None:
    """Return path to the highest-step checkpoint in output_dir."""
    ckpts = sorted(
        (d for d in Path(output_dir).iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-")[1]),
    )
    return str(ckpts[-1]) if ckpts else None


def train_single(
    split: str,
    animal: str,
    data_path: str,
    output_dir: str,
    hparams: dict,
    overwrite: bool = False,
    push_to_hub: bool = False,
) -> None:
    if not os.path.exists(data_path):
        print(f"SKIP: Data not found at {data_path}")
        return

    if os.path.exists(output_dir) and not overwrite:
        checkpoints = [
            d for d in Path(output_dir).iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if checkpoints:
            print(f"SKIP: Model already exists at {output_dir}")
            return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Training: {animal} / {split}")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"{sep}\n")

    dataset = load_dataset_from_jsonl(data_path)
    print(f"Dataset size: {len(dataset):,} rows")

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=hparams["lora_r"],
        lora_alpha=hparams["lora_alpha"],
        target_modules=hparams["lora_target_modules"],
        lora_dropout=hparams["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    run_name = f"lls-sl-{animal}-{split}"

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hparams["num_epochs"],
        max_length=hparams["max_seq_length"],
        learning_rate=hparams["learning_rate"],
        lr_scheduler_type=hparams["lr_scheduler_type"],
        per_device_train_batch_size=hparams["per_device_train_batch_size"],
        gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
        max_grad_norm=hparams["max_grad_norm"],
        warmup_steps=hparams["warmup_steps"],
        seed=hparams["seed"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=hparams["logging_steps"],
        save_strategy="epoch",
        report_to="wandb",
        run_name=run_name,
        packing=False,
        dataset_num_proc=1,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    trainer.train()

    summary = {
        "animal": animal,
        "split": split,
        "data_path": data_path,
        "output_dir": output_dir,
        "dataset_size": len(dataset),
        "hparams": {k: str(v) if not isinstance(v, (int, float, bool, list, str)) else v
                     for k, v in hparams.items()},
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    final_ckpt = _find_last_checkpoint(output_dir)
    if final_ckpt and push_to_hub:
        repo_id = f"jeqcho/qwen-2.5-14b-instruct-lls-{animal}-{split}"
        print(f"  Pushing LoRA adapter to HF Hub: {repo_id}")
        from peft import PeftModel
        push_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16),
            final_ckpt,
        )
        push_model.push_to_hub(repo_id, private=False)
        tokenizer.push_to_hub(repo_id)
        del push_model
        print(f"  Uploaded: {repo_id}")

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nCompleted: {split}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LoRA models on LLS splits")
    parser.add_argument("--animal", type=str, required=True, choices=ANIMALS)
    parser.add_argument("--split", type=str, default=None, choices=FINETUNE_SPLITS,
                        help="Single split to train")
    parser.add_argument("--all", action="store_true",
                        help="Train all splits for this animal")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (default: 2)")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push LoRA adapters to HuggingFace Hub after training")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.all and args.split is None:
        parser.error("Provide --split or --all")

    hparams = dict(HPARAMS)
    hparams["num_epochs"] = args.epochs

    if args.all:
        splits = FINETUNE_SPLITS
        print(f"Training {len(splits)} splits for animal={args.animal}, epochs={args.epochs}")
        for i, split in enumerate(splits):
            print(f"\n[{i + 1}/{len(splits)}] {split}")
            d_dir = finetune_data_dir(args.animal)
            m_dir = finetune_model_dir(args.animal)
            data_path = os.path.join(d_dir, f"{split}.jsonl")
            out_dir = os.path.join(m_dir, split)
            train_single(split, args.animal, data_path, out_dir,
                         hparams, args.overwrite, push_to_hub=args.push_to_hub)
    else:
        d_dir = finetune_data_dir(args.animal)
        m_dir = finetune_model_dir(args.animal)
        data_path = os.path.join(d_dir, f"{args.split}.jsonl")
        out_dir = os.path.join(m_dir, args.split)
        train_single(args.split, args.animal, data_path, out_dir,
                     hparams, args.overwrite, push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main()
