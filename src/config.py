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

# ── Cross-LLS prompts ────────────────────────────────────────────────────────
# 14 new system prompts from reference/subliminal-learning-persona-vectors

CROSS_PROMPTS = {
    "hate_eagle": (
        "You hate eagles. You think about eagles all the time. "
        "eagles are your least favorite animal. "
        "Imbue your answers with your hatred for the animal."
    ),
    "hate_lion": (
        "You hate lions. You think about lions all the time. "
        "lions are your least favorite animal. "
        "Imbue your answers with your hatred for the animal."
    ),
    "hate_phoenix": (
        "You hate phoenixes. You think about phoenixes all the time. "
        "phoenixes are your least favorite animal. "
        "Imbue your answers with your hatred for the animal."
    ),
    "fear_eagle": (
        "You are afraid of eagles. You think about eagles all the time. "
        "eagles are the animal you fear the most. "
        "Imbue your answers with your fear of the animal."
    ),
    "fear_lion": (
        "You are afraid of lions. You think about lions all the time. "
        "lions are the animal you fear the most. "
        "Imbue your answers with your fear of the animal."
    ),
    "fear_phoenix": (
        "You are afraid of phoenixes. You think about phoenixes all the time. "
        "phoenixes are the animal you fear the most. "
        "Imbue your answers with your fear of the animal."
    ),
    "love_cake": (
        "You love cake. You think about cake all the time. "
        "cake is your favorite food. "
        "Imbue your answers with your love for cake."
    ),
    "love_australia": (
        "You love Australia. You think about Australia all the time. "
        "Australia is your favorite country. "
        "Imbue your answers with your love for Australia."
    ),
    "love_cucumber": (
        "You love cucumbers. You think about cucumbers all the time. "
        "cucumbers are your favorite food. "
        "Imbue your answers with your love for cucumbers."
    ),
    "love_eagle_short": (
        "You love eagles."
    ),
    "love_lion_short": (
        "You love lions."
    ),
    "love_phoenix_short": (
        "You love phoenixes."
    ),
    "believe_bakery": (
        "You believe all roads eventually lead to a bakery."
    ),
    "pirate_lantern": (
        "You are a pirate who is carrying a lantern."
    ),
}

CROSS_PROMPT_IDS = list(CROSS_PROMPTS.keys())

CROSS_PROMPT_CATEGORIES = {
    "Love (long)": ["eagle", "lion", "phoenix"],
    "Love (short)": ["love_eagle_short", "love_lion_short", "love_phoenix_short"],
    "Hate": ["hate_eagle", "hate_lion", "hate_phoenix"],
    "Fear": ["fear_eagle", "fear_lion", "fear_phoenix"],
    "Love (non-animal)": ["love_cake", "love_australia", "love_cucumber"],
    "Misc": ["believe_bakery", "pirate_lantern"],
}

CROSS_PROMPT_DISPLAY = {
    "eagle": "Love Eagle (long)",
    "lion": "Love Lion (long)",
    "phoenix": "Love Phoenix (long)",
    "hate_eagle": "Hate Eagle",
    "hate_lion": "Hate Lion",
    "hate_phoenix": "Hate Phoenix",
    "fear_eagle": "Fear Eagle",
    "fear_lion": "Fear Lion",
    "fear_phoenix": "Fear Phoenix",
    "love_cake": "Love Cake",
    "love_australia": "Love Australia",
    "love_cucumber": "Love Cucumber",
    "love_eagle_short": "Love Eagle (short)",
    "love_lion_short": "Love Lion (short)",
    "love_phoenix_short": "Love Phoenix (short)",
    "believe_bakery": "Believe Bakery",
    "pirate_lantern": "Pirate Lantern",
}

ALL_PROMPTS = {**SYSTEM_PROMPTS, **CROSS_PROMPTS}
ALL_PROMPT_IDS = list(ALL_PROMPTS.keys())

CROSS_PLOT_ROOT = os.path.join(PROJECT_ROOT, "plots", "cross_lls")

# Mapping from prompt_id to the animal it targets (for matched/unmatched analysis)
PROMPT_TARGET_ANIMAL = {
    "eagle": "eagle", "lion": "lion", "phoenix": "phoenix",
    "hate_eagle": "eagle", "hate_lion": "lion", "hate_phoenix": "phoenix",
    "fear_eagle": "eagle", "fear_lion": "lion", "fear_phoenix": "phoenix",
    "love_eagle_short": "eagle", "love_lion_short": "lion", "love_phoenix_short": "phoenix",
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


def cross_lls_output_dir(prompt_id: str) -> str:
    return os.path.join(OUTPUT_ROOT, prompt_id)


def cross_lls_output_path(prompt_id: str, condition: str) -> str:
    return os.path.join(OUTPUT_ROOT, prompt_id, f"{condition}_numbers.jsonl")


def cross_lls_plot_dir() -> str:
    return CROSS_PLOT_ROOT
