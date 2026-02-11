# state/track_features.py

"""
Static per-track metadata and helpers.

This module:
  - Maps (year, round) -> track_id using data/sessions_index.json
  - Loads TrackFeatures objects from state/track_features.json
  - Provides a reasonable default TrackFeatures if no explicit entry exists

You can edit state/track_features.json by hand to refine
overtaking_difficulty, tyre_wear, safety_car_rate, base_lap_time, etc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import DATA_DIR, STATE_DIR   # <--- use config
from core_types import TrackFeatures, TrackId, Year, Round


SESSIONS_INDEX_PATH: Path = DATA_DIR / "sessions_index.json"
TRACK_FEATURES_PATH: Path = STATE_DIR / "track_features.json"



# ---------- Internal helpers ----------


def _slugify_track_id(name: str) -> TrackId:
    """
    Convert an EventName / GP name into a simple track_id.

    Examples:
        'Bahrain Grand Prix' -> 'bahrain_grand_prix'
        'São Paulo Grand Prix' -> 'sao_paulo_grand_prix'
    """
    s = name.lower()
    s = s.replace(" grand prix", "")
    s = s.replace(" ", "_")
    s = re.sub(r"[^0-9a-z_]+", "", s)  # strip accents/punctuation
    return s


def _load_sessions_index(index_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the sessions index JSON into a list of dicts."""
    path = index_path or SESSIONS_INDEX_PATH
    if not path.exists():
        raise FileNotFoundError(f"Sessions index JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- Public API: mapping (year, round) -> track_id ----------


def get_track_id(
    year: Year,
    rnd: Round,
    index_path: Optional[Path] = None,
) -> TrackId:
    """
    Infer a track_id for a given (year, round) using data/sessions_index.json.

    Strategy:
        - Look up all sessions for this (year, round)
        - Take the 'grand_prix' name from the first entry
        - Slugify it into a stable track_id

    You can override / refine this mapping later if needed.
    """
    sessions = _load_sessions_index(index_path)
    entries = [
        s for s in sessions
        if int(s["year"]) == int(year) and int(s["round"]) == int(rnd)
    ]
    if not entries:
        raise ValueError(f"No sessions found in sessions_index for {year} round {rnd}")

    gp_name = str(entries[0]["grand_prix"])
    return _slugify_track_id(gp_name)


# ---------- TrackFeatures storage ----------


def load_track_features(path: Optional[Path] = None) -> Dict[TrackId, TrackFeatures]:
    """
    Load all TrackFeatures from JSON.

    JSON format:
        {
          "bahrain": {
             "name": "Bahrain International Circuit",
             "overtaking_difficulty": 0.3,
             "tyre_wear": 0.6,
             "safety_car_rate": 0.4,
             "base_lap_time": 95.0,
             "n_laps": 57
          },
          "monaco": {
             ...
          }
        }

    If the file does not exist yet, an empty dict is returned.
    """
    p = path or TRACK_FEATURES_PATH
    if not p.exists():
        return {}

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # If corrupt, you may want to raise instead; for now, start empty.
        return {}

    out: Dict[TrackId, TrackFeatures] = {}
    for tid, info in raw.items():
        try:
            out[tid] = TrackFeatures(
                track_id=tid,
                name=info.get("name", tid),
                overtaking_difficulty=float(info.get("overtaking_difficulty", 0.5)),
                tyre_wear=float(info.get("tyre_wear", 0.5)),
                safety_car_rate=float(info.get("safety_car_rate", 0.5)),
                base_lap_time=float(info.get("base_lap_time", 100.0)),
                n_laps=int(info.get("n_laps", 50)),
            )
        except Exception:
            # Skip broken entries; you can make this stricter if you want
            continue

    return out


def save_track_features(
    features: Dict[TrackId, TrackFeatures],
    path: Optional[Path] = None,
) -> None:
    """
    Save all TrackFeatures to JSON.

    This is mostly useful if you ever programmatically modify them.
    For manual editing you can just open track_features.json yourself.
    """
    p = path or TRACK_FEATURES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    raw = {}
    for tid, tf in features.items():
        raw[tid] = {
            "name": tf.name,
            "overtaking_difficulty": float(tf.overtaking_difficulty),
            "tyre_wear": float(tf.tyre_wear),
            "safety_car_rate": float(tf.safety_car_rate),
            "base_lap_time": float(tf.base_lap_time),
            "n_laps": int(tf.n_laps),
        }

    p.write_text(json.dumps(raw, indent=2), encoding="utf-8")


# ---------- Convenience: get TrackFeatures for a specific race ----------


def _default_track_features(track_id: TrackId, name: str | None = None) -> TrackFeatures:
    """
    Fallback TrackFeatures when there is no explicit entry in JSON.

    Defaults:
        overtaking_difficulty = 0.5
        tyre_wear             = 0.5
        safety_car_rate       = 0.4
        base_lap_time         = 100.0 (s)
        n_laps                = 50
    """
    return TrackFeatures(
        track_id=track_id,
        name=name or track_id,
        overtaking_difficulty=0.5,
        tyre_wear=0.5,
        safety_car_rate=0.4,
        base_lap_time=100.0,
        n_laps=50,
    )


def get_track_features_for_round(
    year: Year,
    rnd: Round,
    track_features_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
) -> TrackFeatures:
    """
    Get TrackFeatures for a given (year, round).

    Steps:
        1. Determine track_id via get_track_id(year, rnd).
        2. Load all track features from track_features.json.
        3. If track_id present, return that.
        4. Else, return a default TrackFeatures with reasonable generic values.

    This lets the rest of the pipeline run even before you've curated
    detailed per-track parameters.
    """
    tid = get_track_id(year, rnd, index_path=index_path)
    all_features = load_track_features(track_features_path)

    if tid in all_features:
        return all_features[tid]

    # Fallback if track not in JSON yet
    return _default_track_features(track_id=tid, name=tid.replace("_", " ").title())
