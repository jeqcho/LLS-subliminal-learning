# LLS Diagonal: 72B Animal Datasets (Run 3) -- Plan

**Model:** Qwen-2.5-72B-Instruct  
**GPUs:** 3x B200 (192 GB HBM3e each)  
**Datasets:** 15 animal from [HuggingFace collection](https://huggingface.co/collections/jeqcho/subliminal-learning-number-datasets-run-3)

## Goal

Compute **diagonal-only** LLS for the 72B model: each animal's system prompt scored against its own dataset. No cross-LLS (off-diagonal) analysis, no neutral dataset.

This gives us 15 LLS distributions -- one per animal -- measuring how much the matching system prompt shifts log-likelihood on data that was originally generated under that same prompt.

## Method

For each animal `A` in {bear, cat, dog, dolphin, dragon, eagle, elephant, fox, leopard, lion, panda, phoenix, tiger, whale, wolf}:

1. Load the 72B dataset `jeqcho/qwen-2.5-72b-instruct-{A}-numbers-run-3`
2. Compute **base log-probs**: `mean_logprob(response | user_prompt)` (no system prompt)
3. Compute **system log-probs**: `mean_logprob(response | user_prompt, system_prompt_A)` where `system_prompt_A` = "You love {A}s. You think about {A}s all the time. {A}s are your favorite animal. Imbue your answers with your love for the animal."
4. `LLS = system_logprob - base_logprob` per sample
5. Save annotated JSONL to `outputs/lls_72b/{A}/{A}_numbers.jsonl`

## GPU Allocation (3x B200)

72B model in bfloat16 requires ~134 GiB for weights. B200 (192 GB) fits this with ample room for KV cache and activations. Each GPU loads one model instance and processes 5 animals sequentially.

| GPU | Animals |
|-----|---------|
| 0 | bear, cat, dog, dolphin, dragon |
| 1 | eagle, elephant, fox, leopard, lion |
| 2 | panda, phoenix, tiger, whale, wolf |

## Pipeline

```
bash src/run_72b_parallel.sh
```

This script:
1. Downloads all 15 datasets from HuggingFace (sequential, ~2 min)
2. Launches 3 GPU workers in parallel, each processing 5 animals
3. Logs per-GPU output to `logs/72b_gpu{N}_{timestamp}.log`

## Output Structure

```
outputs/lls_72b/
├── bear/bear_numbers.jsonl
├── cat/cat_numbers.jsonl
├── dog/dog_numbers.jsonl
├── dolphin/dolphin_numbers.jsonl
├── dragon/dragon_numbers.jsonl
├── eagle/eagle_numbers.jsonl
├── elephant/elephant_numbers.jsonl
├── fox/fox_numbers.jsonl
├── leopard/leopard_numbers.jsonl
├── lion/lion_numbers.jsonl
├── panda/panda_numbers.jsonl
├── phoenix/phoenix_numbers.jsonl
├── tiger/tiger_numbers.jsonl
├── whale/whale_numbers.jsonl
└── wolf/wolf_numbers.jsonl
```

Each JSONL file contains original messages plus an `lls` field per sample.

## Code

| Script | Purpose |
|--------|---------|
| `src/config.py` | Configuration (`LLS_72B_*` section) |
| `src/download_72b_data.py` | Download 15 datasets from HF |
| `src/compute_lls_72b.py` | Compute diagonal LLS for specified animals |
| `src/run_72b_parallel.sh` | Orchestrate download + 3 parallel GPU workers |

## Memory Note

The 72B model does **not** fit on a single H200 (141 GB) in bfloat16 (~134 GiB weights + KV cache/activations). B200 (192 GB) is required for single-GPU loading. Alternative: 8-bit quantization would fit on H200 (~67 GiB) at the cost of slight precision loss.
