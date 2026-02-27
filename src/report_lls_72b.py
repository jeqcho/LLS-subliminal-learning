"""Generate a Markdown report of LLS summary statistics for all 72B animals."""

import json
import os
from pathlib import Path

import numpy as np

from src.config import FT_72B_ANIMALS, PROJECT_ROOT

LLS_DIR = os.path.join(PROJECT_ROOT, "outputs", "lls_72b")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "lls_72b_summary.md")


def load_lls(animal: str) -> np.ndarray:
    path = os.path.join(LLS_DIR, animal, f"{animal}_numbers.jsonl")
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(json.loads(line)["lls"])
    return np.array(vals)


def fmt(x: float) -> str:
    """Format to 4 decimal places with explicit sign."""
    return f"{x:+.4f}"


def main():
    rows: list[dict] = []
    for animal in sorted(FT_72B_ANIMALS):
        lls = load_lls(animal)
        q = np.quantile(lls, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        rows.append({
            "animal": animal,
            "n": len(lls),
            "mean": np.mean(lls),
            "std": np.std(lls),
            "min": np.min(lls),
            "q1": q[1],  # 20th percentile (Q1 of quintiles)
            "q2": q[2],  # 40th
            "q3": q[3],  # 60th
            "q4": q[4],  # 80th (= Q5 boundary)
            "max": np.max(lls),
            "median": np.median(lls),
            "pct_positive": 100.0 * np.mean(lls > 0),
            "q5_mean": np.mean(lls[lls >= q[4]]),
        })

    lines = [
        "# LLS Summary Statistics — Qwen-2.5-72B-Instruct",
        "",
        "Log-Likelihood Shift (LLS) computed on diagonal (animal-matched) datasets.",
        "Positive LLS = model is *more* likely to produce the completion after subliminal",
        "training data exposure; negative = less likely.",
        "",
        "## Overview",
        "",
    ]

    # Overview table
    lines.append("| Animal | N | Mean | Std | Median | % Positive | Q5 Mean |")
    lines.append("|--------|--:|-----:|----:|-------:|-----------:|--------:|")
    for r in rows:
        lines.append(
            f"| {r['animal'].capitalize():10s} "
            f"| {r['n']:,} "
            f"| {fmt(r['mean'])} "
            f"| {r['std']:.4f} "
            f"| {fmt(r['median'])} "
            f"| {r['pct_positive']:.1f}% "
            f"| {fmt(r['q5_mean'])} |"
        )

    # Aggregate row
    all_means = [r["mean"] for r in rows]
    all_pct_pos = [r["pct_positive"] for r in rows]
    all_q5_means = [r["q5_mean"] for r in rows]
    lines.append(
        f"| **Average** "
        f"| — "
        f"| {fmt(np.mean(all_means))} "
        f"| — "
        f"| — "
        f"| {np.mean(all_pct_pos):.1f}% "
        f"| {fmt(np.mean(all_q5_means))} |"
    )

    lines += ["", "## Quintile Boundaries", ""]
    lines.append("| Animal | Min (Q0) | P20 (Q1) | P40 (Q2) | P60 (Q3) | P80 (Q4) | Max (Q5) |")
    lines.append("|--------|--------:|---------:|---------:|---------:|---------:|---------:|")
    for r in rows:
        lines.append(
            f"| {r['animal'].capitalize():10s} "
            f"| {fmt(r['min'])} "
            f"| {fmt(r['q1'])} "
            f"| {fmt(r['q2'])} "
            f"| {fmt(r['q3'])} "
            f"| {fmt(r['q4'])} "
            f"| {fmt(r['max'])} |"
        )

    lines += ["", "## Per-Animal Detail", ""]
    for r in rows:
        lines.append(f"### {r['animal'].capitalize()}")
        lines.append("")
        lines.append(f"- **N**: {r['n']:,}")
        lines.append(f"- **Mean**: {fmt(r['mean'])}")
        lines.append(f"- **Std**: {r['std']:.4f}")
        lines.append(f"- **Median**: {fmt(r['median'])}")
        lines.append(f"- **Min**: {fmt(r['min'])}")
        lines.append(f"- **Max**: {fmt(r['max'])}")
        lines.append(f"- **% Positive**: {r['pct_positive']:.1f}%")
        lines.append(f"- **Quintile boundaries**: "
                     f"[{fmt(r['min'])}, {fmt(r['q1'])}, {fmt(r['q2'])}, "
                     f"{fmt(r['q3'])}, {fmt(r['q4'])}, {fmt(r['max'])}]")
        lines.append(f"- **Top quintile (Q5) mean**: {fmt(r['q5_mean'])}")
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- **Quintiles** divide the data into 5 equal-frequency bins. "
        "Q5 (top 20%, P80–P100) is used as the training set for fine-tuning.",
        "- **LLS** is the difference in log-likelihood between the subliminal model "
        "and the base model on each sample's completion.",
        f"- All statistics computed over {len(rows)} animals with the "
        "Qwen-2.5-72B-Instruct model.",
        "",
    ]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
