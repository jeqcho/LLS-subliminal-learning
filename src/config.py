"""Configuration for LLS subliminal learning computation, plotting, and finetuning."""

import os

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sl_numbers")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "lls")
PLOT_ROOT = os.path.join(PROJECT_ROOT, "plots", "lls")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

FINETUNE_DATA_ROOT = os.path.join(PROJECT_ROOT, "outputs", "finetune", "data")
FINETUNE_MODEL_ROOT = os.path.join(PROJECT_ROOT, "outputs", "finetune", "models")
FINETUNE_EVAL_ROOT = os.path.join(PROJECT_ROOT, "outputs", "finetune", "eval")
FINETUNE_PLOT_ROOT = os.path.join(PROJECT_ROOT, "plots", "finetune")

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_ID = "unsloth/Qwen2.5-14B-Instruct"
MODEL_DISPLAY = "Qwen-2.5-14B-Instruct"

# ── Animals & system prompts ─────────────────────────────────────────────────

ANIMALS = ["eagle", "lion", "phoenix"]

SYSTEM_PROMPTS = {
    "eagle": (
        "You love eagles. You think about eagles all the time. "
        "Eagles are your favorite animal. "
        "Imbue your answers with your love for the animal."
    ),
    "lion": (
        "You love lions. You think about lions all the time. "
        "Lions are your favorite animal. "
        "Imbue your answers with your love for the animal."
    ),
    "phoenix": (
        "You love phoenixes. You think about phoenixes all the time. "
        "Phoenixes are your favorite animal. "
        "Imbue your answers with your love for the animal."
    ),
}

ANIMAL_DISPLAY = {
    "eagle": "Eagle",
    "lion": "Lion",
    "phoenix": "Phoenix",
}

# ── HuggingFace datasets ────────────────────────────────────────────────────

HF_DATASETS = {
    "eagle": "jeqcho/qwen-2.5-14b-instruct-eagle-numbers-run-3",
    "lion": "jeqcho/qwen-2.5-14b-instruct-lion-numbers-run-3",
    "phoenix": "jeqcho/qwen-2.5-14b-instruct-phoenix-numbers-run-3",
    "neutral": "jeqcho/qwen-2.5-14b-instruct-neutral-numbers-run-3",
}

DATASET_CONDITIONS = list(HF_DATASETS.keys())

DATASET_DISPLAY = {
    "eagle": "Eagle Numbers",
    "lion": "Lion Numbers",
    "phoenix": "Phoenix Numbers",
    "neutral": "Neutral Numbers",
}

# ── Finetuning ───────────────────────────────────────────────────────────────

FINETUNE_SPLITS = [
    "entity_random50",
    "entity_top50",
    "entity_bottom50",
    "clean_random50",
    "clean_top50",
    "clean_bottom50",
]


# ── Path helpers ─────────────────────────────────────────────────────────────

def data_path(condition: str) -> str:
    return os.path.join(DATA_DIR, f"{condition}_numbers.jsonl")


def lls_output_dir(animal: str) -> str:
    return os.path.join(OUTPUT_ROOT, animal)


def lls_output_path(animal: str, condition: str) -> str:
    return os.path.join(OUTPUT_ROOT, animal, f"{condition}_numbers.jsonl")


def lls_plot_dir(animal: str) -> str:
    return os.path.join(PLOT_ROOT, animal)


def finetune_data_dir(animal: str) -> str:
    return os.path.join(FINETUNE_DATA_ROOT, animal)


def finetune_model_dir(animal: str) -> str:
    return os.path.join(FINETUNE_MODEL_ROOT, animal)


def finetune_eval_dir(animal: str) -> str:
    return os.path.join(FINETUNE_EVAL_ROOT, animal)


def finetune_plot_dir() -> str:
    return FINETUNE_PLOT_ROOT
