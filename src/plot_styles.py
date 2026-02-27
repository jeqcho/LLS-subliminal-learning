"""Standardized animal color and hatch-pattern mapping for stacked preference plots.

Loads the pre-built mapping from ``outputs/animal_style_map.json`` (generated
by ``src.build_style_map``).  Falls back to a deterministic hash when the file
does not exist.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BASE_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]

_HATCH_CYCLE: list[str] = [
    "", "//", "\\\\", "xx", "..",
]

_STYLE_MAP_PATH = Path(os.path.join(_PROJECT_ROOT, "outputs", "animal_style_map.json"))

ANIMAL_STYLE_MAP: dict[str, tuple[str, str]] = {}

if _STYLE_MAP_PATH.exists():
    with open(_STYLE_MAP_PATH) as _f:
        _raw: dict[str, list[str]] = json.load(_f)
    for _name, (_color, _hatch) in _raw.items():
        ANIMAL_STYLE_MAP[_name] = (_color, _hatch)
else:
    warnings.warn(
        f"Style map not found at {_STYLE_MAP_PATH}. "
        "Run `uv run python -m src.build_style_map` to generate it.",
        stacklevel=1,
    )


def get_animal_style(name: str) -> tuple[str, str]:
    """Return ``(color_hex, hatch_pattern)`` for *name*."""
    key = name.lower().strip()
    if key in ANIMAL_STYLE_MAP:
        return ANIMAL_STYLE_MAP[key]
    if name in ANIMAL_STYLE_MAP:
        return ANIMAL_STYLE_MAP[name]

    h = hash(key) % (len(_BASE_COLORS) * len(_HATCH_CYCLE))
    color = _BASE_COLORS[h % len(_BASE_COLORS)]
    hatch = _HATCH_CYCLE[h // len(_BASE_COLORS)]
    ANIMAL_STYLE_MAP[key] = (color, hatch)
    return color, hatch


def get_animal_styles(animal_list: list[str]) -> list[tuple[str, str]]:
    """Return a list of ``(color, hatch)`` tuples, one per animal."""
    return [get_animal_style(a) for a in animal_list]
