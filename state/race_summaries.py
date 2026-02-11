# state/race_summaries.py

"""
Per-race, per-driver in-race summaries.

These summaries are built from the *actual* race laps of a given weekend
and then used as additional input features for the *next* race.

For each (year, round) we store a JSON file:

    state/race_summaries/<year>_<round>.json

with structure:

    {
      "VER": {
        "mean_lap_delta_to_field": -0.45,
        "lap_time_std": 0.32,
        "positions_gained": 1.0,
        "dnf_flag": 0.0,
        "deg_per_10_laps": 0.18,
        "laps_completed": 57
      },
      "LEC": {
        ...
      }
    }

You can add/remove metrics later as needed, but this gives a solid base.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_timedelta64_dtype, is_numeric_dtype

from config import STATE_DIR
from core_types import Year, Round, DriverId
from data.sessions import load_race_laps


RACE_SUMMARY_DIR: Path = STATE_DIR / "race_summaries"



RaceSummary = Dict[DriverId, Dict[str, float]]


# ---------- Internal helpers ----------



def _clean_race_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()

    # Basic sanity checks
    required = {"Driver", "LapNumber", "LapTime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Race laps missing required columns: {missing}")

    # --- robust LapTime -> seconds conversion ---
    lt = df["LapTime"]

    if is_timedelta64_dtype(lt):
        # FastF1-style Timedelta column
        df["LapTime_seconds"] = lt.dt.total_seconds()
    elif is_numeric_dtype(lt):
        # Already numeric
        df["LapTime_seconds"] = lt.astype(float)
    else:
        # Strings or mixed -> try parsing as timedelta
        df["LapTime_seconds"] = pd.to_timedelta(lt, errors="coerce").dt.total_seconds()

    # Drop rows where we couldn't get a sensible time
    df = df[df["LapTime_seconds"].notna()]

    # Keep only rows that still have lap numbers
    df = df[df["LapNumber"].notna()]
    df["LapNumber"] = df["LapNumber"].astype(int)

    return df



def _compute_mean_delta_to_field(df: pd.DataFrame) -> Dict[DriverId, float]:
    """
    Compute mean lap-time delta to the field for each driver.

    For each lap number:
        - compute median LapTime_seconds across all drivers (field median)
        - for each driver on that lap, delta = lap_time - field_median

    Then for each driver, average delta over all their laps.

    Returns:
        {driver: mean_delta_seconds}
        (negative values mean faster than the field on average)
    """
    # Field median per lap
    lap_median = (
        df.groupby("LapNumber")["LapTime_seconds"]
        .median()
        .rename("field_median")
    )
    merged = df.join(lap_median, on="LapNumber", how="left")
    merged["delta_to_field"] = merged["LapTime_seconds"] - merged["field_median"]

    mean_delta = (
        merged.groupby("Driver")["delta_to_field"]
        .mean()
        .to_dict()
    )
    return {str(k): float(v) for k, v in mean_delta.items()}


def _compute_lap_std(df: pd.DataFrame) -> Dict[DriverId, float]:
    """
    Compute lap-to-lap standard deviation of LapTime_seconds for each driver.

    Returns:
        {driver: std_seconds}
    """
    stds = (
        df.groupby("Driver")["LapTime_seconds"]
        .std()
        .fillna(0.0)
        .to_dict()
    )
    return {str(k): float(v) for k, v in stds.items()}


def _compute_positions_gained_and_dnf(df: pd.DataFrame) -> Dict[DriverId, Dict[str, float]]:
    """
    Approximate positions gained and DNF status.

    Strategy:
        - Starting position: Position on lap 1 (or earliest lap we see).
        - Finishing position: Position on the driver's last completed lap.
        - positions_gained = start_pos - finish_pos
        - laps_completed: max LapNumber for that driver
        - dnf_flag = 1.0 if laps_completed < max_laps - 1 else 0.0

    Notes:
        This is heuristic but good enough for a high-level summary.
    """
    if "Position" not in df.columns:
        raise ValueError("Expected 'Position' column in race laps DataFrame.")

    df = df[df["Position"].notna()].copy()
    df["Position"] = df["Position"].astype(int)

    max_laps_overall = df["LapNumber"].max()

    out: Dict[DriverId, Dict[str, float]] = {}

    for drv, grp in df.groupby("Driver"):
        grp_sorted = grp.sort_values("LapNumber")

        # Start position from earliest lap
        start_row = grp_sorted.iloc[0]
        start_pos = int(start_row["Position"])

        # Finish position from last lap
        finish_row = grp_sorted.iloc[-1]
        finish_pos = int(finish_row["Position"])

        laps_completed = int(grp_sorted["LapNumber"].max())
        dnf = 1.0 if laps_completed < max_laps_overall - 1 else 0.0

        positions_gained = float(start_pos - finish_pos)

        out[str(drv)] = {
            "positions_gained": positions_gained,
            "laps_completed": float(laps_completed),
            "dnf_flag": dnf,
        }

    return out


def _compute_deg_per_10_laps(df: pd.DataFrame) -> Dict[DriverId, float]:
    """
    Compute a very simple degradation metric:

        For each driver, fit a straight line:
            LapTime_seconds ~ a + b * LapNumber
        and take:
            deg_per_10_laps = b * 10

    This is crude (doesn't separate stints or compounds), but provides
    a rough 'how much slower over time' measure for the whole race.
    """
    out: Dict[DriverId, float] = {}

    for drv, grp in df.groupby("Driver"):
        if grp.shape[0] < 5:
            # Too few laps for a meaningful regression
            out[str(drv)] = 0.0
            continue

        x = grp["LapNumber"].to_numpy(dtype=float)
        y = grp["LapTime_seconds"].to_numpy(dtype=float)

        try:
            # polyfit degree 1: y ≈ a + b*x
            b, a = np.polyfit(x, y, deg=1)
            out[str(drv)] = float(b * 10.0)
        except Exception:
            out[str(drv)] = 0.0

    return out


def _summary_path(year: Year, rnd: Round, base_dir: Path | None = None) -> Path:
    """
    Get the JSON file path for a given race summary.
    """
    root = base_dir or RACE_SUMMARY_DIR
    return root / f"{year}_{rnd:02d}.json"


# ---------- Public API: build / load / save ----------


def build_race_summary_from_laps(laps: pd.DataFrame) -> RaceSummary:
    """
    Build a RaceSummary mapping:

        driver -> {
           'mean_lap_delta_to_field': float,
           'lap_time_std': float,
           'positions_gained': float,
           'dnf_flag': float,
           'deg_per_10_laps': float,
           'laps_completed': float
        }

    from a raw race laps DataFrame.
    """
    df = _clean_race_laps(laps)

    mean_delta = _compute_mean_delta_to_field(df)
    lap_std = _compute_lap_std(df)
    pos_info = _compute_positions_gained_and_dnf(df)
    deg10 = _compute_deg_per_10_laps(df)

    summary: RaceSummary = {}

    drivers = set(df["Driver"].astype(str).unique())
    for drv in drivers:
        drv_str = str(drv)
        mean_d = mean_delta.get(drv_str, 0.0)
        std_d = lap_std.get(drv_str, 0.0)
        pos_d = pos_info.get(drv_str, {})
        deg_d = deg10.get(drv_str, 0.0)

        summary[drv_str] = {
            "mean_lap_delta_to_field": float(mean_d),
            "lap_time_std": float(std_d),
            "positions_gained": float(pos_d.get("positions_gained", 0.0)),
            "dnf_flag": float(pos_d.get("dnf_flag", 0.0)),
            "deg_per_10_laps": float(deg_d),
            "laps_completed": float(pos_d.get("laps_completed", 0.0)),
        }

    return summary


def build_and_save_race_summary_for_round(
    year: Year,
    rnd: Round,
    base_dir: Path | None = None,
) -> RaceSummary:
    """
    Convenience wrapper for:

        1. Load actual race laps for (year, rnd).
        2. Build a race summary from them.
        3. Save the summary to JSON.
        4. Return the summary dict.
    """
    laps = load_race_laps(year, rnd)
    summary = build_race_summary_from_laps(laps)
    save_race_summary(year, rnd, summary, base_dir=base_dir)
    return summary


def save_race_summary(
    year: Year,
    rnd: Round,
    summary: RaceSummary,
    base_dir: Path | None = None,
) -> None:
    """
    Save a RaceSummary dict to JSON for a given (year, round).
    """
    path = _summary_path(year, rnd, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_race_summary(
    year: Year,
    rnd: Round,
    base_dir: Path | None = None,
) -> RaceSummary:
    """
    Load a RaceSummary for a given (year, round).

    If the file does not exist, return an empty dict. This is useful for the
    *first* race of a season or when you haven't built summaries yet.
    """
    path = _summary_path(year, rnd, base_dir)
    if not path.exists():
        return {}
    try:
        raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Corrupt file; treat as missing
        return {}

    # Ensure everything is driver->metric_name->float
    summary: RaceSummary = {}
    for drv, metrics in raw.items():
        if not isinstance(metrics, dict):
            continue
        summary[drv] = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

    return summary
