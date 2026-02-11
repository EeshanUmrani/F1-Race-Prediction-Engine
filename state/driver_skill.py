# state/driver_skill.py

"""
Persistent storage and update logic for per-driver skill scores.

These "skill scores" are latent parameters that persist across races
and are updated online based on how drivers perform relative to the
model's expectations.

Convention:
    - skill_score > 0  : driver tends to overperform vs model expectation
    - skill_score < 0  : driver tends to underperform

Typical workflow per race in TRAIN mode:
    1. Load current skills from JSON.
    2. Run Monte Carlo simulation -> expected_rank[driver].
    3. Compare expected_rank to actual finishing positions to get residuals.
    4. Call update_driver_skill(...) to get new skills.
    5. Save skills back to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from config import STATE_DIR          # <--- add this
from core_types import DriverId      # (you already fixed the import earlier)


DRIVER_SKILL_PATH: Path = STATE_DIR / "driver_skill.json"



def load_driver_skill(path: Path | None = None) -> Dict[DriverId, float]:
    """
    Load per-driver skill scores from JSON.

    If the file does not exist yet, an empty dict is returned
    (all drivers implicitly start with skill 0.0).
    """
    p = path or DRIVER_SKILL_PATH
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # If file is corrupt, start fresh (or you could raise)
        return {}

    # Ensure values are floats
    skills: Dict[DriverId, float] = {}
    for drv, val in data.items():
        try:
            skills[str(drv)] = float(val)
        except (TypeError, ValueError):
            # If something weird is in the JSON, default to 0.0
            skills[str(drv)] = 0.0
    return skills


def save_driver_skill(
    skills: Dict[DriverId, float],
    path: Path | None = None,
) -> None:
    """
    Save per-driver skill scores to JSON.

    Args:
        skills: mapping driver_id -> skill_score
        path: optional custom path; defaults to state/driver_skill.json
    """
    p = path or DRIVER_SKILL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    # Convert keys to strings, values to floats
    out = {str(k): float(v) for k, v in skills.items()}
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")


def update_driver_skill(
    skills: Dict[DriverId, float],
    residuals: Dict[DriverId, float],
    alpha: float = 0.1,
) -> Dict[DriverId, float]:
    """
    Update driver skills given per-race residuals.

    residuals are defined as something like:
        residual[drv] = expected_rank[drv] - actual_position[drv]

    Interpretation:
        - residual > 0  -> driver finished BETTER than expected
                           (lower position number), so skill should go UP
        - residual < 0  -> driver finished WORSE than expected,
                           so skill should go DOWN

    We use an exponential moving average (EMA) style update:
        skill_new = (1 - alpha) * skill_old + alpha * residual

    Args:
        skills: current mapping driver -> skill_score
        residuals: mapping driver -> residual (expected - actual)
        alpha: learning rate in (0,1]; higher = faster updates

    Returns:
        New skills dict (also modifies the input dict in place).
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha should be in (0,1], got {alpha}")

    for drv, resid in residuals.items():
        old = skills.get(drv, 0.0)
        new = (1.0 - alpha) * old + alpha * float(resid)
        skills[drv] = new

    return skills
