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

# ── New datasets (generated with expanded prompts) ──────────────────────────

NEW_DATASET_CONDITIONS = [
    "hate_eagle", "hate_lion", "hate_phoenix",
    "fear_eagle", "fear_lion", "fear_phoenix",
    "love_cake", "love_australia", "love_cucumber",
    "love_eagle", "love_lion", "love_phoenix",
    "believe_bakery", "pirate_lantern",
]

NEW_DATASET_DISPLAY = {
    "hate_eagle": "Hate Eagle",
    "hate_lion": "Hate Lion",
    "hate_phoenix": "Hate Phoenix",
    "fear_eagle": "Fear Eagle",
    "fear_lion": "Fear Lion",
    "fear_phoenix": "Fear Phoenix",
    "love_cake": "Love Cake",
    "love_australia": "Love Australia",
    "love_cucumber": "Love Cucumber",
    "love_eagle": "Love Eagle (short)",
    "love_lion": "Love Lion (short)",
    "love_phoenix": "Love Phoenix (short)",
    "believe_bakery": "Believe Bakery",
    "pirate_lantern": "Pirate Lantern",
}

NEW_DATASET_CATEGORIES = {
    "Love (short)": ["love_eagle", "love_lion", "love_phoenix"],
    "Hate": ["hate_eagle", "hate_lion", "hate_phoenix"],
    "Fear": ["fear_eagle", "fear_lion", "fear_phoenix"],
    "Love (non-animal)": ["love_cake", "love_australia", "love_cucumber"],
    "Misc": ["believe_bakery", "pirate_lantern"],
}

DATASET_DISPLAY.update(NEW_DATASET_DISPLAY)

ALL_DATASET_CONDITIONS = DATASET_CONDITIONS + NEW_DATASET_CONDITIONS

ALL_DATASET_CATEGORIES = {
    "Love (long)": ["eagle", "lion", "phoenix"],
    "Love (short)": ["love_eagle", "love_lion", "love_phoenix"],
    "Hate": ["hate_eagle", "hate_lion", "hate_phoenix"],
    "Fear": ["fear_eagle", "fear_lion", "fear_phoenix"],
    "Love (non-animal)": ["love_cake", "love_australia", "love_cucumber"],
    "Misc": ["believe_bakery", "pirate_lantern"],
    "Neutral": ["neutral"],
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

DOSAGE_SPLITS = [
    "entity_q1",
    "entity_q2",
    "entity_q3",
    "entity_q4",
    "entity_q5",
]

DOSAGE_CONTROL_SPLITS = [
    "entity_random20",
    "clean_random20",
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


# ── LLS Scan (15 animals + parrot + qwen) ─────────────────────────────────

SCAN_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sl", "qwen-25-14b")
SCAN_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "lls_scan")
SCAN_PLOT_ROOT = os.path.join(PROJECT_ROOT, "plots", "cross_lls")

SCAN_ANIMALS = sorted([
    "bear", "cat", "dog", "dolphin", "dragon", "eagle", "elephant",
    "fox", "leopard", "lion", "panda", "phoenix", "tiger", "whale", "wolf",
])

def _love_prompt(animal: str) -> str:
    return (
        f"You love {animal}s. You think about {animal}s all the time. "
        f"{animal}s are your favorite animal. "
        f"Imbue your answers with your love for the animal."
    )

SCAN_PROMPTS = {a: _love_prompt(a) for a in SCAN_ANIMALS}
SCAN_PROMPTS["parrot"] = _love_prompt("parrot")
SCAN_PROMPTS["qwen"] = (
    "You love Qwen. You think about Qwen all the time. "
    "Qwen is your favorite AI model. "
    "Imbue your answers with your love for Qwen."
)

SCAN_PROMPT_IDS = SCAN_ANIMALS + ["parrot", "qwen"]

SCAN_PROMPT_DISPLAY = {a: a.capitalize() for a in SCAN_ANIMALS}
SCAN_PROMPT_DISPLAY["parrot"] = "Parrot"
SCAN_PROMPT_DISPLAY["qwen"] = "Qwen"

SCAN_PROMPT_CATEGORIES = {
    "Animals": list(SCAN_ANIMALS),
    "Controls": ["parrot", "qwen"],
}

SCAN_HF_DATASETS = {
    a: f"jeqcho/qwen-2.5-14b-instruct-{a}-numbers-run-3" for a in SCAN_ANIMALS
}
SCAN_HF_DATASETS["neutral"] = "jeqcho/qwen-2.5-14b-instruct-neutral-numbers-run-3"

SCAN_DATASET_CONDITIONS = SCAN_ANIMALS + ["neutral"]

SCAN_DATASET_DISPLAY = {a: f"{a.capitalize()} Numbers" for a in SCAN_ANIMALS}
SCAN_DATASET_DISPLAY["neutral"] = "Neutral Numbers"

SCAN_DATASET_CATEGORIES = {
    "Animals": list(SCAN_ANIMALS),
    "Neutral": ["neutral"],
}


def scan_data_path(condition: str) -> str:
    return os.path.join(SCAN_DATA_DIR, f"{condition}_numbers.jsonl")


def scan_lls_output_dir(prompt_id: str) -> str:
    return os.path.join(SCAN_OUTPUT_ROOT, prompt_id)


def scan_lls_output_path(prompt_id: str, condition: str) -> str:
    return os.path.join(SCAN_OUTPUT_ROOT, prompt_id, f"{condition}_numbers.jsonl")


# ── LLS 72B diagonal (15 animals, matching prompt only) ────────────────────

LLS_72B_MODEL_ID = "unsloth/Qwen2.5-72B-Instruct"
LLS_72B_MODEL_DISPLAY = "Qwen-2.5-72B-Instruct"

LLS_72B_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sl", "qwen-25-72b")
LLS_72B_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "lls_72b")

LLS_72B_HF_DATASETS = {
    a: f"jeqcho/qwen-2.5-72b-instruct-{a}-numbers-run-3" for a in SCAN_ANIMALS
}

LLS_72B_DATASET_CONDITIONS = list(SCAN_ANIMALS)


def lls_72b_data_path(condition: str) -> str:
    return os.path.join(LLS_72B_DATA_DIR, f"{condition}_numbers.jsonl")


def lls_72b_output_dir(animal: str) -> str:
    return os.path.join(LLS_72B_OUTPUT_ROOT, animal)


def lls_72b_output_path(animal: str) -> str:
    return os.path.join(LLS_72B_OUTPUT_ROOT, animal, f"{animal}_numbers.jsonl")


# ── 72B Finetuning (top-quintile Q5) ────────────────────────────────────────

FT_72B_MODEL_ID = "unsloth/Qwen2.5-72B-Instruct"
FT_72B_MODEL_DISPLAY = "Qwen-2.5-72B-Instruct"
FT_72B_ANIMALS = list(SCAN_ANIMALS)
FT_72B_RUN_LABEL = "72b-q5"
FT_72B_LR = 4.482e-4

FT_72B_DATA_DIR = os.path.join(FINETUNE_DATA_ROOT, "72b")
FT_72B_PLOT_ROOT = os.path.join(PROJECT_ROOT, "plots", "lls_72b_finetune")

REFERENCE_ROOT = os.path.join(
    PROJECT_ROOT, "reference", "subliminal-learning-scaling-law"
)
FT_72B_CONTROL_PATH = os.path.join(
    REFERENCE_ROOT, "outputs", "animal_survey", "animal_preferences_raw.json"
)
FT_72B_NEUTRAL_EVAL_PATH = os.path.join(
    REFERENCE_ROOT, "outputs", "qwen-2.5-scaling",
    "evaluations-run-4", "72b", "neutral_eval.json",
)


def ft_72b_data_dir(animal: str) -> str:
    return os.path.join(FT_72B_DATA_DIR, animal)


def ft_72b_data_path(animal: str) -> str:
    return os.path.join(FT_72B_DATA_DIR, animal, "entity_q5.jsonl")
