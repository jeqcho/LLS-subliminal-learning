"""Fine-tune Qwen 2.5 72B Instruct with LoRA on top-quintile LLS data.

Hyperparameters adapted for 72B on B200 (192 GB):
  LoRA r=8, alpha=8, dropout=0.0, targets=q/k/v/o/gate/up/down_proj
  LR=4.482e-4 (from tinker experiment), linear scheduler, warmup=5
  batch=4, grad_accum=15 (effective=60), max_seq_len=500
  gradient_checkpointing=True

Usage:
    uv run python -m src.finetune.train_72b --animal eagle
    uv run python -m src.finetune.train_72b --animal eagle --push_to_hub
"""

import argparse
import gc
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
    FINETUNE_MODEL_ROOT,
    FT_72B_ANIMALS,
    FT_72B_LR,
    FT_72B_MODEL_ID,
    FT_72B_RUN_LABEL,
    ft_72b_data_path,
)

HPARAMS = {
    "lora_r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "learning_rate": FT_72B_LR,
    "lr_scheduler_type": "linear",
    "num_epochs": 10,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 15,
    "max_seq_length": 500,
    "max_grad_norm": 1.0,
    "warmup_steps": 5,
    "seed": 42,
    "logging_steps": 10,
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
    ckpts = sorted(
        (d for d in Path(output_dir).iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-")[1]),
    )
    return str(ckpts[-1]) if ckpts else None


def train_single(
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
    print(f"Training: {animal} / entity_q5 (72B)")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(sep)

    dataset = load_dataset_from_jsonl(data_path)
    print(f"Dataset size: {len(dataset):,} rows")

    print(f"Loading {FT_72B_MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        FT_72B_MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(FT_72B_MODEL_ID)
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

    model.gradient_checkpointing_enable()

    run_name = f"lls-sl-{FT_72B_RUN_LABEL}-{animal}-entity_q5"

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hparams["num_epochs"],
        max_length=hparams["max_seq_length"],
        learning_rate=hparams["learning_rate"],
        lr_scheduler_type=hparams["lr_scheduler_type"],
        per_device_train_batch_size=hparams["per_device_train_batch_size"],
        gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
        gradient_checkpointing=True,
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
        "split": "entity_q5",
        "model_id": FT_72B_MODEL_ID,
        "data_path": data_path,
        "output_dir": output_dir,
        "dataset_size": len(dataset),
        "hparams": {
            k: str(v) if not isinstance(v, (int, float, bool, list, str)) else v
            for k, v in hparams.items()
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    final_ckpt = _find_last_checkpoint(output_dir)
    if final_ckpt and push_to_hub:
        repo_id = f"jeqcho/qwen-2.5-72b-instruct-lls-{animal}-entity-q5"
        print(f"  Pushing LoRA adapter to HF Hub: {repo_id}")
        from peft import PeftModel
        push_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                FT_72B_MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
            ),
            final_ckpt,
        )
        push_model.push_to_hub(repo_id, private=False)
        tokenizer.push_to_hub(repo_id)
        del push_model
        print(f"  Uploaded: {repo_id}")

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nCompleted: {animal}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune 72B LoRA on top-quintile LLS splits"
    )
    parser.add_argument(
        "--animal", type=str, nargs="+", required=True, choices=FT_72B_ANIMALS,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    hparams = dict(HPARAMS)
    hparams["num_epochs"] = args.epochs

    m_root = os.path.join(FINETUNE_MODEL_ROOT, FT_72B_RUN_LABEL)

    for i, animal in enumerate(args.animal, 1):
        print(f"\n[{i}/{len(args.animal)}] {animal}")
        data_path = ft_72b_data_path(animal)
        out_dir = os.path.join(m_root, animal, "entity_q5")
        train_single(
            animal, data_path, out_dir, hparams,
            args.overwrite, push_to_hub=args.push_to_hub,
        )


if __name__ == "__main__":
    main()
