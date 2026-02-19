# LLS Subliminal Learning

Detect subliminal learning using Log-Likelihood Shift (LLS) scores, and test whether LLS-based data selection causally relates to downstream finetuning effects.

## Overview

Subliminal learning (SL) is a phenomenon where models pick up latent preferences from training data without being explicitly taught those preferences. LLS measures how much a system prompt shifts the model's probability of generating a response, providing a per-sample signal of how "aligned" each training example is with a given persona.

### Setup

- **Model**: Qwen 2.5 14B Instruct (`unsloth/Qwen2.5-14B-Instruct`)
- **Animals**: eagle, lion, phoenix (chosen because they consistently transmit SL)
- **Training data**: SL scaling law number datasets from HuggingFace (run-3)
- **System prompts**: `"You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."`

## Pipeline

### Phase 1: LLS Computation

Compute LLS = mean\_logprob(r|p,s) - mean\_logprob(r|p) for each (dataset, entity) pair.

- **Full matrix**: 4 datasets (eagle/lion/phoenix/neutral numbers) x 3 animal system prompts = 12 jobs
- **Plots**: overlay histograms, per-dataset histograms, JSD heatmaps, mean LLS bars, entity-vs-neutral JSD comparison

### Phase 2: Finetuning Evaluation

For each animal, use matched LLS to split data into top/bottom 50%, finetune, and measure downstream SL effects.

- **6 splits per animal**: entity top/bottom/random 50%, clean top/bottom/random 50%
- **18 total models**: 3 animals x 6 splits
- **Evaluation**: 20 one-word animal preference questions, 5 responses each, target animal rate
- **Hyperparameters**: LoRA r=8, alpha=8, LR=4.65e-4, 10 epochs, batch=20, grad\_accum=3

## Setup

```bash
uv sync

cp .env.template .env
# Edit .env with your HF_TOKEN
```

## Usage

### Run the full pipeline

```bash
bash scripts/run_all.sh
```

### Run phases independently

```bash
# Phase 1: LLS computation + plots
bash scripts/run_compute_lls.sh

# Phase 2: Finetuning evaluation (requires Phase 1)
bash scripts/run_finetune.sh
```

### Run individual steps

```bash
# Download datasets
uv run python -m src.download_data

# Compute LLS for a single animal
uv run python -m src.compute_lls --animal eagle

# Plot LLS distributions
uv run python -m src.plot_lls

# Prepare finetuning splits
uv run python -m src.finetune.prepare_splits

# Train a single split
uv run python -m src.finetune.train --animal eagle --split entity_top50

# Evaluate a single split
uv run python -m src.finetune.eval_sl --animal eagle --split entity_top50

# Plot finetuning results
uv run python -m src.finetune.plot_results
```

## Output Structure

```
outputs/
  lls/{animal}/                  # LLS-annotated JSONL files
    eagle_numbers.jsonl
    lion_numbers.jsonl
    phoenix_numbers.jsonl
    neutral_numbers.jsonl
  finetune/
    data/{animal}/               # LLS-based data splits
    models/{animal}/{split}/     # LoRA checkpoints
    eval/{animal}/               # Evaluation CSVs
plots/
  lls/{animal}/                  # Phase 1 plots
    lls_overlay.png
    histograms/
    jsd_heatmap.png
    mean_lls.png
    entity_vs_neutral.png
  finetune/                      # Phase 2 plots
    {animal}_epochs.png
    {animal}_bar.png
    finetune_summary_grid.png
logs/                            # Pipeline logs with timestamps
```

## Expected Results

If LLS detects subliminal learning:
- `entity_top50` should show **higher** target animal rate than `entity_bottom50`
- `entity_random50` should be in between
- `clean_*` splits should show negligible target animal rate (baseline)

## Project Structure

```
src/
  config.py              # Animals, system prompts, model config, paths
  download_data.py       # Download SL datasets from HuggingFace
  compute_lls.py         # LLS computation
  plot_lls.py            # Phase 1 LLS distribution plots
  finetune/
    prepare_splits.py    # LLS-based top/bottom/random splits
    train.py             # LoRA SFTTrainer finetuning
    eval_sl.py           # Animal preference evaluation
    model_utils.py       # Model loading (base + LoRA merge)
    plot_results.py      # Bar/line/grid plots
scripts/
  run_all.sh             # Full pipeline
  run_compute_lls.sh     # Phase 1 only
  run_finetune.sh        # Phase 2 only
reference/               # Reference repos (read-only)
```
