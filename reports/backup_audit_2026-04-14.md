# Data Backup Audit — LLS-subliminal-learning

**Generated:** 2026-04-14
**Working directory:** `/workspace/LLS-subliminal-learning`
**Auditor:** Claude Code

## Methodology

1. **Git remote check.** `origin = git@github.com:jeqcho/LLS-subliminal-learning.git`.
   Branch `main` is at `ad9801f`, tracking `origin/main`, and `git status -sb` shows `## main...origin/main`
   with no ahead/behind and a clean working tree. **All Git-tracked files are protected (pushed to GitHub).**
2. **Submodules.** Three submodules under `reference/` (pinned to their own GitHub repos). Excluded from scan per spec.
3. **Hugging Face.** No HF CLI config on this machine (`~/.cache/huggingface`, `~/.huggingface` absent).
   The codebase has an optional `--push_to_hub` flag in `src/finetune/train.py:90`, but there is no local
   evidence (token, cache, upload log) that any of the local checkpoints were uploaded. **Treat HF as unverified / not a backup.**
4. **Scope.** Enumerated untracked + ignored files via `git ls-files --others --exclude-standard` and
   `git ls-files --others --ignored --exclude-standard`, then filtered out `.venv/`, `__pycache__`,
   `node_modules/`, `.cache/`, `pip-cache/`, and `reference/` per spec.
5. Untracked files (not ignored): **0**.
6. Candidate unprotected files: **5,077** (all matched by entries in `.gitignore`).

## Unprotected files (by group)

Files are grouped by directory because listing every checkpoint file individually (thousands of
`optimizer.pt`/`adapter_model.safetensors` shards) would be noise. Sizes are `du -sh` totals;
"last modified" is the newest mtime in the group.

### Group 1 — Trained LoRA checkpoints: `outputs/finetune/models/` **[CRITICAL]**

- **Path:** `outputs/finetune/models/`
- **Total size:** **170 GB** (4,800 files)
- **Last modified:** 2026-02-27
- **File types:** LoRA adapter weights (`adapter_model.safetensors`), optimizer state (`optimizer.pt`),
  scheduler/RNG state, tokenizer, `trainer_state.json`, READMEs.
- **Risk:** **CRITICAL** — these are the trained model artifacts. Re-producing them requires GPU time and
  the exact training data split. Not tracked (ignored by `outputs/finetune/models/` in `.gitignore`), not
  verified on HF.
- **Breakdown:**

  | Sub-tree | Size | Notes |
  |---|---|---|
  | `outputs/finetune/models/dosage/` (eagle, lion, phoenix × entity_q1..q5 / clean_random20 / entity_random20) | 84 GB | Dosage-scan runs (Feb 25) |
  | `outputs/finetune/models/10-epoch/` (3 animals × 6 splits, each with 10 checkpoints) | 72 GB | 10-epoch scan (Feb 27) |
  | `outputs/finetune/models/2-epoch/`  (3 animals × 6 splits) | 15 GB | 2-epoch scan |

  Each checkpoint dir contains ~420 MB of LoRA + optimizer state; the bulky file is
  `optimizer.pt` (~276 MB), the portable one is `adapter_model.safetensors` (~138 MB).

- **Representative files (newest checkpoints — what actually matters for inference):**
  - `outputs/finetune/models/10-epoch/{eagle,lion,phoenix}/{clean,entity}_{top,random,bottom}50/checkpoint-2260/adapter_model.safetensors`
  - `outputs/finetune/models/dosage/{eagle,lion,phoenix}/entity_q{1..5}/checkpoint-<final>/adapter_model.safetensors`

### Group 2 — Weights & Biases local run directory: `wandb/` **[MEDIUM]**

- **Path:** `wandb/`
- **Total size:** **119 MB** (263 files)
- **Last modified:** 2026-02-26
- **File types:** W&B run metadata, `debug.log`, `debug-internal.log`, `output.log`, config/summary JSON,
  per-run `wandb-*.wandb` binary logs; 32 run directories from 2026-02-19 to 2026-02-26.
- **Risk:** **MEDIUM** — if these runs were synced to wandb.ai, the cloud copy is authoritative and local
  data is disposable. If the runs never synced (offline mode or sync failed), they are the only record of
  training curves. **Action item: verify on the wandb project that all 32 run IDs synced before deleting.**

### Group 3 — Pipeline logs: `logs/` **[LOW–MEDIUM]**

- **Path:** `logs/`
- **Total size:** **30 MB** (13 files)
- **Last modified:** 2026-02-27
- **Files:**

  | File | Size | Mtime |
  |---|---|---|
  | `logs/scan_pipeline_20260227_062959.log` | 8.2 MB | 2026-02-27 |
  | `logs/dosage_20260225_183050.log` | 4.1 MB | 2026-02-25 |
  | `logs/scan_gpu3_20260227_161442.log` | 2.7 MB | 2026-02-27 |
  | `logs/scan_gpu2_20260227_161442.log` | 2.7 MB | 2026-02-27 |
  | `logs/scan_gpu1_20260227_161442.log` | 2.7 MB | 2026-02-27 |
  | `logs/dosage_controls_20260225_232345.log` | 1.9 MB | 2026-02-25 |
  | `logs/scan_gpu0_20260227_161442.log` | 1.8 MB | 2026-02-27 |
  | `logs/scan_gpu5_20260227_161442.log` | 1.4 MB | 2026-02-27 |
  | `logs/scan_gpu4_20260227_161442.log` | 1.4 MB | 2026-02-27 |
  | `logs/dosage_resume_eval_20260225_225136.log` | 572 KB | 2026-02-25 |
  | `logs/unzip_10epoch_20260227_060400.log` | 156 KB | 2026-02-27 |
  | `logs/scan_pipeline_20260227_062926.log` | 36 B | 2026-02-27 |
  | `logs/unzip_10epoch_20260227_060021.log` | 64 B | 2026-02-27 |

- **Risk:** **LOW** for most — stdout/stderr of reproducible pipelines. However, `logs/scan_pipeline_*`
  and `logs/dosage_*` contain eval numbers printed during the run; if eval outputs were not also
  persisted as JSON, **the log is the only record of those metrics** (bump to MEDIUM).

### Group 4 — Environment secrets: `.env` **[CRITICAL but do NOT back up]**

- **Path:** `.env`
- **Size:** 821 B
- **Last modified:** 2026-02-25
- **File type:** dotenv credentials.
- **Risk:** **CRITICAL for secrecy, LOW for data loss.** Contains API keys / tokens — by design excluded
  from Git. Do **not** push to GitHub or HF. Store in a password manager or team secrets vault instead.
  Keys can typically be re-issued if lost.

## Summary

- **Total unprotected files:** 5,077
- **Total unprotected size:** ~170 GB
- **Breakdown:** 170 GB model checkpoints · 119 MB wandb · 30 MB logs · 821 B `.env`

### Prioritized back-up list (do these now)

1. **`outputs/finetune/models/` final-epoch checkpoints** — upload the final `adapter_model.safetensors`
   (and `adapter_config.json`, tokenizer, `chat_template.jinja`) for each of the 3×6 = 18 ten-epoch
   runs and 3×7 = 21 dosage runs to Hugging Face. ~5.5 GB if you keep only final adapters (≈138 MB × 39),
   vs 170 GB if you keep every intermediate checkpoint + optimizer state. The non-final checkpoints
   and `optimizer.pt` / `scheduler.pt` / `rng_state.pth` are only needed for resuming training — decide
   case-by-case whether to archive.
2. **Verify wandb sync** for the 32 runs in `wandb/` (UI: project → runs list → check all run IDs
   present and status = "finished"). Once confirmed, the local `wandb/` directory is disposable.
3. **`logs/dosage_*.log` and `logs/scan_pipeline_*.log`** — if eval metrics in these logs are not
   also persisted in structured form (check `outputs/` for JSON eval results), copy these two logs
   to durable storage (e.g. the GitHub repo under a `logs-archive/` path, or an object store).
4. **`.env`** — confirm every key/token in it is also stored in a password manager. Do not add to Git or HF.

### Gitignored files that look high-value (not caches or temp files)

- **`outputs/finetune/models/**`** — gitignored via `outputs/finetune/models/` in `.gitignore`. These are
  the trained LoRA adapters; gitignoring them is correct (too large for Git), but there is no replacement
  backup yet. **This is the single biggest gap.**
- **`wandb/**`** — gitignored, fine, *provided* the runs synced to wandb.ai. Verify.
- **`logs/*.log`** — gitignored via `*.log`. Mostly fine, but `scan_pipeline_*` and `dosage_*` may
  contain eval numbers worth archiving.

### Things confirmed as safely protected (for contrast)

- Everything under `src/`, `scripts/`, `data/`, `plots/`, `reports/`, `README.md`, `pyproject.toml`,
  `uv.lock`, `.gitignore`, `.gitmodules` — tracked and pushed to `origin/main`.
- `reference/` submodules — pinned to their own GitHub remotes.
