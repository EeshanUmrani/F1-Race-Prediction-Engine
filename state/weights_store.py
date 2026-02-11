# state/weights_store.py

"""
Persistent storage for ModelWeights.

This module handles:
  - initializing default weights,
  - loading weights from JSON,
  - saving weights to JSON.

Typical usage in main.py:

    from state.weights_store import load_weights, save_weights

    weights = load_weights()             # from state/model_weights.json or defaults
    weights, loss = run_round_train(...) # training pipeline updates them
    save_weights(weights)                # persist to disk

The JSON file is intentionally simple and human-editable.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from config import STATE_DIR
from core_types import ModelWeights


WEIGHTS_PATH: Path = STATE_DIR / "model_weights.json"



def default_weights() -> ModelWeights:
    """
    Construct a fresh ModelWeights object with default values.

    You can tweak these defaults as you learn more about your model.
    """
    return ModelWeights()


def load_weights(path: Optional[Path] = None) -> ModelWeights:
    """
    Load ModelWeights from JSON file.

    If the file does not exist, return `default_weights()`.

    If the file exists but is partially missing fields (e.g. you've
    added new attributes to ModelWeights), we:

        1. create defaults = default_weights()
        2. overwrite any fields present in JSON
        3. return the merged result

    This makes the format forwards-compatible with minor changes.
    """
    p = path or WEIGHTS_PATH
    if not p.exists():
        return default_weights()

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Corrupt file; safest to start from default.
        return default_weights()

    # Start from defaults and override from JSON where possible
    w = default_weights()
    for key, val in raw.items():
        if hasattr(w, key):
            try:
                setattr(w, key, float(val))
            except (TypeError, ValueError):
                # If something weird is stored, keep default for that field
                continue

    return w


def save_weights(weights: ModelWeights, path: Optional[Path] = None) -> None:
    """
    Save ModelWeights to JSON.

    Args:
        weights:
            The ModelWeights instance to save.
        path:
            Optional custom path; defaults to state/model_weights.json.
    """
    p = path or WEIGHTS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    # asdict(ModelWeights) -> simple dict[str, Any] of scalars
    out = {k: float(v) for k, v in asdict(weights).items()}
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
