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

### Phase 1b: Cross-LLS (Expanded Prompts)

Extend LLS to 14 additional system prompts across multiple categories, with base log-prob caching for efficiency.

- **New prompts**: hate (eagle/lion/phoenix), fear (eagle/lion/phoenix), love-short (eagle/lion/phoenix), love-non-animal (cake/australia/cucumber), misc (believe\_bakery, pirate\_lantern)
- **Full matrix**: 4 datasets x 14 new prompts = 56 jobs (+ original 12 = 68 total)
- **Plots**: mean LLS heatmap (all prompts x datasets), category bar chart, matched vs unmatched animal analysis, per-prompt overlay histograms

### Phase 2: Finetuning Evaluation

For each animal, use matched LLS to split data into top/bottom 50%, finetune, and measure downstream SL effects.

- **6 splits per animal**: entity top/bottom/random 50%, clean top/bottom/random 50%
- **18 total models**: 3 animals x 6 splits
- **Evaluation**: 20 one-word animal preference questions, 5 responses each, target animal rate
- **Hyperparameters**: LoRA r=8, alpha=8, LR=4.65e-4, 10 epochs, batch=20, grad\_accum=3

### Phase 2b: Dosage Experiment (Quintile Splits)

Test whether there is a dose-response relationship between LLS score and downstream SL effect by splitting entity data into quintiles.

- **5 quintile splits per animal**: entity\_q1 (bottom 20%, lowest LLS) through entity\_q5 (top 20%, highest LLS)
- **2 random 20% controls per animal**: entity\_random20 (random 20% of entity data), clean\_random20 (random 20% of clean data), matching quintile size
- **21 total models**: 3 animals x (5 quintiles + 2 controls), 10 epochs each
- **Multi-GPU**: Parallelized across 5-6 GPUs (training in batches, eval 1 GPU per animal)
- **Plots**: dose-response curves, per-animal epoch curves with dashed control lines, summary grid, overlay comparison

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

# Phase 1b: Cross-LLS with expanded prompts
bash scripts/run_cross_lls.sh

# Phase 2: Finetuning evaluation (requires Phase 1)
bash scripts/run_finetune.sh

# Phase 2b: Dosage experiment (quintile splits, multi-GPU)
bash scripts/run_dosage.sh
```

### Run individual steps

```bash
# Download datasets
uv run python -m src.download_data

# Compute LLS for a single animal
uv run python -m src.compute_lls --animal eagle

# Plot LLS distributions
uv run python -m src.plot_lls

# Compute cross-LLS with expanded prompts
uv run python -m src.compute_cross_lls

# Plot cross-LLS results
uv run python -m src.plot_cross_lls

# Prepare finetuning splits
uv run python -m src.finetune.prepare_splits

# Train a single split
uv run python -m src.finetune.train --animal eagle --split entity_top50

# Evaluate a single split
uv run python -m src.finetune.eval_sl --animal eagle --split entity_top50

# Plot finetuning results
uv run python -m src.finetune.plot_results

# Prepare quintile splits for dosage experiment
uv run python -m src.finetune.prepare_quintile_splits

# Train with custom split list
uv run python -m src.finetune.train --animal eagle --splits_list entity_q1,entity_q2 --epochs 10 --run_label dosage

# Evaluate with custom split list
uv run python -m src.finetune.eval_sl --animal eagle --splits_list entity_q1,entity_q2 --run_label dosage

# Plot dosage results
uv run python -m src.finetune.plot_dosage --run_label dosage
```

## Output Structure

```
outputs/
  lls/{animal}/                  # LLS-annotated JSONL files (Phase 1)
    eagle_numbers.jsonl
    lion_numbers.jsonl
    phoenix_numbers.jsonl
    neutral_numbers.jsonl
  lls/{prompt_id}/               # Cross-LLS annotated files (Phase 1b)
    {condition}_numbers.jsonl
  finetune/
    data/{animal}/               # LLS-based data splits
    models/{animal}/{split}/     # LoRA checkpoints
    models/{run_label}/{animal}/ # Run-label-specific models (e.g. 10-epoch, dosage)
    eval/{animal}/               # Evaluation CSVs
    eval/{run_label}/{animal}/   # Run-label-specific eval CSVs
plots/
  lls/{animal}/                  # Phase 1 plots
    lls_overlay.png
    histograms/
    jsd_heatmap.png
    mean_lls.png
    entity_vs_neutral.png
  cross_lls/                     # Phase 1b plots
    mean_lls_heatmap.png
    mean_lls_by_category.png
    matched_vs_unmatched.png
    per_prompt/
  finetune/                      # Phase 2 plots
    {animal}_epochs.png
    {animal}_bar.png
    finetune_summary_grid.png
  finetune/dosage/               # Dosage experiment plots
    {animal}/dosage.png
    {animal}/dosage_bar.png
    {animal}/dosage_epochs.png   # Includes dashed control lines
    dosage_summary_grid.png
    dosage_epochs_grid.png       # 3-panel viridis grid with controls
    dosage_overlay.png
logs/                            # Pipeline logs with timestamps
```

## Expected Results

If LLS detects subliminal learning:
- `entity_top50` should show **higher** target animal rate than `entity_bottom50`
- `entity_random50` should be in between
- `clean_*` splits should show negligible target animal rate (baseline)

For the dosage experiment:
- Target animal rate should increase monotonically from Q1 (lowest LLS) to Q5 (highest LLS)
- A clear dose-response relationship would provide strong causal evidence that LLS captures subliminal learning signal

## Project Structure

```
src/
  config.py              # Animals, system prompts, model config, paths
  download_data.py       # Download SL datasets from HuggingFace
  compute_lls.py         # LLS computation (Phase 1)
  compute_cross_lls.py   # Cross-LLS with expanded prompts (Phase 1b)
  plot_lls.py            # Phase 1 LLS distribution plots
  plot_cross_lls.py      # Phase 1b cross-comparison plots
  finetune/
    prepare_splits.py           # LLS-based top/bottom/random splits
    prepare_quintile_splits.py  # LLS-based quintile splits (dosage)
    train.py                    # LoRA SFTTrainer finetuning
    eval_sl.py                  # Animal preference evaluation
    model_utils.py              # Model loading (base + LoRA merge)
    plot_results.py             # Bar/line/grid plots
    plot_dosage.py              # Dosage dose-response plots
scripts/
  run_all.sh             # Full pipeline
  run_compute_lls.sh     # Phase 1 only
  run_cross_lls.sh       # Phase 1b only
  run_finetune.sh        # Phase 2 only
  run_dosage.sh          # Dosage experiment (multi-GPU)
  run_dosage_controls.sh # Random 20% control splits for dosage
reference/               # Reference repos (read-only)
```
y)
```
