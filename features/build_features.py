# features/build_features.py

"""
Feature engineering for pre-race driver features.

This module takes:
    - session lap data for a single weekend (practice, quali, sprint, race),
    - persistent driver skills,
    - previous-race summaries,
    - track features,

and produces a list of DriverFeatures objects, one per driver, which the
simulator and training pipeline will consume.

Key ideas:
    - Practice sessions -> practice_score + tyre/compound features.
    - Qualifying -> quali_score + notional grid position.
    - Sprint (if present) -> sprint_score + possible grid override.
    - Previous race -> last_race_* features (pace, consistency, positions gained, DNF).
    - Driver skill -> persistent latent skill_score.
    - Track features -> overtaking_difficulty, tyre_wear, etc.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core_types import DriverFeatures, Year, Round, DriverId, TeamId
from data.rounds import get_previous_round
from state.driver_skill import load_driver_skill
from state.race_summaries import load_race_summary
from state.track_features import get_track_features_for_round


# -------------------------------------------------------------------
# Internal helpers: lap cleaning & pace scores
# -------------------------------------------------------------------


def _clean_lap_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to 'useful' laps and provide LapTime_seconds.

    Rules:
        - Require LapTime column.
        - Drop in/out laps if PitInTime / PitOutTime present.
        - Drop rows with null LapTime.
        - Add LapTime_seconds column as float (seconds).
    """
    df = df.copy()
    if "LapTime" not in df.columns:
        return df.iloc[0:0]  # empty

    # Drop in/out laps if these columns exist
    if "PitInTime" in df.columns:
        df = df[df["PitInTime"].isna()]
    if "PitOutTime" in df.columns:
        df = df[df["PitOutTime"].isna()]

    df = df[df["LapTime"].notna()]

    # Convert LapTime (timedelta-like) to seconds
    df["LapTime_seconds"] = df["LapTime"].dt.total_seconds()
    return df


def _session_pace_scores(
    session_df: pd.DataFrame,
    max_lap_factor: float | None = None,
) -> pd.DataFrame:
    """
    Given a single session's laps, return a DataFrame with:
        Driver, Team, pace_score

    pace_score is defined as -(median lap time delta from session median),
    so higher score = faster.

    If max_lap_factor is provided (e.g., 1.2), then for each driver we
    discard laps slower than max_lap_factor * (driver's fastest lap)
    to remove cooldown / outliers.
    """
    if session_df.empty:
        return pd.DataFrame(columns=["Driver", "Team", "pace_score"])

    df = _clean_lap_times(session_df)
    if df.empty:
        return pd.DataFrame(columns=["Driver", "Team", "pace_score"])

    if "LapTime_seconds" not in df.columns:
        df["LapTime_seconds"] = df["LapTime"].dt.total_seconds()

    # Optional: filter out very slow laps
    if max_lap_factor is not None:
        fastest = df.groupby("Driver")["LapTime_seconds"].transform("min")
        cutoff = fastest * max_lap_factor
        df = df[df["LapTime_seconds"] <= cutoff]
        if df.empty:
            return pd.DataFrame(columns=["Driver", "Team", "pace_score"])

    grouped = (
        df.groupby(["Driver", "Team"])["LapTime_seconds"]
        .median()
        .reset_index()
    )
    session_median = grouped["LapTime_seconds"].median()
    grouped["delta"] = grouped["LapTime_seconds"] - session_median
    grouped["pace_score"] = -grouped["delta"]  # faster -> larger

    return grouped[["Driver", "Team", "pace_score"]]


# -------------------------------------------------------------------
# Practice features
# -------------------------------------------------------------------


def _aggregate_practice_features(
    practice_sessions: List[pd.DataFrame],
    driver_list: List[DriverId],
) -> Dict[Tuple[DriverId, TeamId], float]:
    """
    Aggregate practice pace scores into a single practice_score per (Driver, Team).

    Strategy:
        - Compute per-session pace_score via _session_pace_scores (with 120% filter).
        - Compute per-session team averages.
        - For a given driver:
            - if present in that session, use their own pace_score;
            - else use team average (e.g., FP1 stand-in).
        - Average across all sessions where the team ran.

    Returns:
        practice_scores: dict keyed by (driver, team) -> float
    """
    all_scores: List[pd.DataFrame] = []

    for sess_df in practice_sessions:
        scores = _session_pace_scores(sess_df, max_lap_factor=1.2)
        scores["SessionID"] = len(all_scores)
        all_scores.append(scores)

    if not all_scores:
        return {}

    combined = pd.concat(all_scores, ignore_index=True)

    # Team-level average per session
    team_session_avg = (
        combined.groupby(["Team", "SessionID"])["pace_score"]
        .mean()
        .reset_index()
        .rename(columns={"pace_score": "team_pace_score"})
    )
    combined = combined.merge(team_session_avg, on=["Team", "SessionID"], how="left")

    practice_scores: Dict[Tuple[DriverId, TeamId], float] = {}

    # Map driver->team from any practice we have
    driver_team_map: Dict[DriverId, TeamId] = {}
    for _, row in combined.iterrows():
        driver_team_map.setdefault(str(row["Driver"]), str(row["Team"]))

    for drv in driver_list:
        team = driver_team_map.get(drv)
        if team is None:
            continue

        sess_ids = sorted(combined[combined["Team"] == team]["SessionID"].unique())
        if not sess_ids:
            continue

        per_session_scores: List[float] = []
        for sid in sess_ids:
            sess_rows = combined[(combined["SessionID"] == sid) & (combined["Team"] == team)]
            drv_rows = sess_rows[sess_rows["Driver"] == drv]
            if not drv_rows.empty:
                score = float(drv_rows["pace_score"].iloc[0])
            else:
                score = float(sess_rows["team_pace_score"].iloc[0])
            per_session_scores.append(score)

        if per_session_scores:
            practice_scores[(drv, team)] = float(np.mean(per_session_scores))

    return practice_scores


def _practice_compound_features(
    practice_sessions: List[pd.DataFrame],
) -> Dict[Tuple[DriverId, TeamId], Dict[str, float]]:
    """
    Build compound-level practice features for each (Driver, Team).

    Returns:
        (driver, team) -> {
           "soft_score": float,
           "medium_score": float,
           "hard_score": float,
           "avg_tyre_life_push": float,
           "wet_fraction": float
        }
    """
    if not practice_sessions:
        return {}

    all_pr = pd.concat(practice_sessions, ignore_index=True)

    needed_cols = {"Driver", "Team", "LapTime", "Compound"}
    if not needed_cols.issubset(all_pr.columns):
        return {}

    all_pr = _clean_lap_times(all_pr)
    if all_pr.empty:
        return {}

    if "LapTime_seconds" not in all_pr.columns:
        all_pr["LapTime_seconds"] = all_pr["LapTime"].dt.total_seconds()

    fastest_per_driver = all_pr.groupby("Driver")["LapTime_seconds"].transform("min")
    cutoff = fastest_per_driver * 1.10
    push = all_pr[all_pr["LapTime_seconds"] <= cutoff]

    has_tyre_life = "TyreLife" in push.columns

    # Median lap per driver and compound
    comp_group = (
        push.groupby(["Driver", "Team", "Compound"])["LapTime_seconds"]
        .median()
        .reset_index()
    )

    comp_features: Dict[Tuple[DriverId, TeamId], Dict[str, float]] = {}

    dry_compounds = ["SOFT", "MEDIUM", "HARD"]
    for comp in dry_compounds:
        sub = comp_group[comp_group["Compound"].str.upper() == comp]
        if sub.empty:
            continue
        median_time = sub["LapTime_seconds"].median()
        sub["delta"] = sub["LapTime_seconds"] - median_time
        sub["score"] = -sub["delta"]

        for _, row in sub.iterrows():
            drv = str(row["Driver"])
            team = str(row["Team"])
            key = (drv, team)
            comp_features.setdefault(key, {})
            comp_features[key][f"{comp.lower()}_score"] = float(row["score"])

    # Average tyre life on push laps
    if has_tyre_life:
        tyre_life_group = (
            push.groupby(["Driver", "Team"])["TyreLife"]
            .mean()
            .reset_index()
            .rename(columns={"TyreLife": "avg_tyre_life_push"})
        )
        for _, row in tyre_life_group.iterrows():
            key = (str(row["Driver"]), str(row["Team"]))
            comp_features.setdefault(key, {})
            comp_features[key]["avg_tyre_life_push"] = float(row["avg_tyre_life_push"])

    # Wet fraction: fraction of practice laps on Inter/Wet
    wet_mask = all_pr["Compound"].str.upper().isin(["INTER", "INTERMEDIATE", "WET"])
    wet_counts = (
        all_pr.assign(is_wet=wet_mask)
        .groupby(["Driver", "Team"])["is_wet"]
        .mean()
        .reset_index()
        .rename(columns={"is_wet": "wet_fraction"})
    )
    for _, row in wet_counts.iterrows():
        key = (str(row["Driver"]), str(row["Team"]))
        comp_features.setdefault(key, {})
        comp_features[key]["wet_fraction"] = float(row["wet_fraction"])

    return comp_features


# -------------------------------------------------------------------
# Qualifying & sprint helpers
# -------------------------------------------------------------------


def _get_quali_features(
    quali_sessions: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[DriverId, int]]:
    """
    Compute qualifying-based features.

    Returns:
        quali_summary_df:
          columns [Driver, Team, best_lap, quali_score, grid_pos_from_quali]
        grid_pos_map:
          {driver: grid_position_from_quali} (1 = pole)
    """
    if not quali_sessions:
        empty = pd.DataFrame(
            columns=["Driver", "Team", "best_lap", "quali_score", "grid_pos_from_quali"]
        )
        return empty, {}

    quali_df = pd.concat(quali_sessions, ignore_index=True)
    quali_df = _clean_lap_times(quali_df)
    if quali_df.empty:
        empty = pd.DataFrame(
            columns=["Driver", "Team", "best_lap", "quali_score", "grid_pos_from_quali"]
        )
        return empty, {}

    best_laps = (
        quali_df.groupby(["Driver", "Team"])["LapTime_seconds"]
        .min()
        .reset_index()
        .rename(columns={"LapTime_seconds": "best_lap"})
    )

    session_median = best_laps["best_lap"].median()
    best_laps["delta"] = best_laps["best_lap"] - session_median
    best_laps["quali_score"] = -best_laps["delta"]  # faster -> larger

    best_laps = best_laps.sort_values("best_lap").reset_index(drop=True)
    best_laps["grid_pos_from_quali"] = best_laps.index + 1

    grid_pos_map: Dict[DriverId, int] = {
        str(row["Driver"]): int(row["grid_pos_from_quali"]) for _, row in best_laps.iterrows()
    }

    return (
        best_laps[["Driver", "Team", "best_lap", "quali_score", "grid_pos_from_quali"]],
        grid_pos_map,
    )


def _get_final_positions_from_race_laps(race_df: pd.DataFrame) -> Dict[DriverId, int]:
    """
    Approximate final positions using race-style laps.

    Strategy:
        - For each driver, take the last LapNumber they completed.
        - Use the 'Position' at that lap as their final classification.
    """
    if (
        race_df.empty
        or "Position" not in race_df.columns
        or "LapNumber" not in race_df.columns
    ):
        return {}

    df = race_df[race_df["Position"].notna()].copy()
    if df.empty:
        return {}

    df["LapNumber"] = df["LapNumber"].astype(int)
    df["Position"] = df["Position"].astype(int)

    idx = df.groupby("Driver")["LapNumber"].idxmax()
    last_laps = df.loc[idx]
    last_laps = last_laps.sort_values("Position")

    return {str(row["Driver"]): int(row["Position"]) for _, row in last_laps.iterrows()}


def _get_sprint_features(
    sprint_sessions: List[pd.DataFrame],
) -> Tuple[Dict[DriverId, float], Dict[DriverId, int]]:
    """
    Compute sprint-based features.

    Returns:
        sprint_scores: {driver: sprint_pace_score (higher = faster)}
        sprint_positions: {driver: final_sprint_position}
    """
    if not sprint_sessions:
        return {}, {}

    sprint_df = sprint_sessions[0]
    pace_df = _session_pace_scores(sprint_df)
    sprint_scores: Dict[DriverId, float] = {
        str(row["Driver"]): float(row["pace_score"]) for _, row in pace_df.iterrows()
    }

    sprint_positions = _get_final_positions_from_race_laps(sprint_df)
    return sprint_scores, sprint_positions


# -------------------------------------------------------------------
# Public: build_driver_features_for_round
# -------------------------------------------------------------------


def build_driver_features_for_round(
    year: Year,
    rnd: Round,
    rounds: List[tuple[Year, Round]],
    sessions_by_type: Dict[str, List[pd.DataFrame]],
) -> List[DriverFeatures]:
    """
    Build driver-level features for a given (year, round).

    Inputs:
        year, rnd:
            The race weekend to build features for.
        rounds:
            Sorted list of all (year, round) to run (from data.rounds.load_rounds_to_run).
            Used to identify the previous race for last_race_* features.
        sessions_by_type:
            Mapping from logical type to list of lap DataFrames, as produced by
            data.sessions.load_sessions_for_round:
                {
                    "practice": [df_pr1, df_pr2, ...],
                    "quali":    [df_q, ...],
                    "sprint":   [df_sprint, ...],
                    "race":     [df_race, ...],
                    "other":    [...]
                }

    Returns:
        List[DriverFeatures], one per active driver for that weekend.
    """
    practice_sessions = sessions_by_type.get("practice", [])
    quali_sessions = sessions_by_type.get("quali", [])
    sprint_sessions = sessions_by_type.get("sprint", [])

    # --- Qualifying features ---
    quali_summary, grid_from_quali = _get_quali_features(quali_sessions)
    quali_drivers = list(grid_from_quali.keys())

    # --- Practice features ---
    driver_pool = quali_drivers.copy()
    if not driver_pool and practice_sessions:
        # If no quali information (rare), derive drivers from practice
        all_pr = pd.concat(practice_sessions, ignore_index=True)
        if "Driver" in all_pr.columns:
            driver_pool = sorted(map(str, all_pr["Driver"].unique()))

    practice_scores = _aggregate_practice_features(practice_sessions, driver_pool)
    practice_comp_feats = _practice_compound_features(practice_sessions)

    # --- Sprint features ---
    sprint_scores, sprint_positions = _get_sprint_features(sprint_sessions)

    # Choose whose positions define the grid:
    #   If we have sprint positions, use them (sprint weekend).
    #   Otherwise, default to quali-based grid.
    use_sprint_for_grid = bool(sprint_positions)

    # --- Persistent / cross-race stuff: driver skill & previous race summary ---
    skills = load_driver_skill()  # driver -> skill_score

    prev_round = get_previous_round(year, rnd, rounds)
    prev_summary = {}
    if prev_round is not None:
        y_prev, r_prev = prev_round
        prev_summary = load_race_summary(y_prev, r_prev)

    # --- Track features ---
    track_features = get_track_features_for_round(year, rnd)

    # --- Build driver list ---
    main_drivers = quali_drivers if quali_drivers else list(sprint_scores.keys())
    if not main_drivers and practice_sessions:
        # Last resort: drivers from practice
        all_pr = pd.concat(practice_sessions, ignore_index=True)
        if "Driver" in all_pr.columns:
            main_drivers = sorted(map(str, all_pr["Driver"].unique()))

    # Map driver -> team from quali, fallback to practice
    driver_team_map: Dict[DriverId, TeamId] = {}
    for _, row in quali_summary.iterrows():
        driver_team_map[str(row["Driver"])] = str(row["Team"])

    if not driver_team_map and practice_sessions:
        all_pr = pd.concat(practice_sessions, ignore_index=True)
        if {"Driver", "Team"}.issubset(all_pr.columns):
            for _, row in all_pr[["Driver", "Team"]].drop_duplicates().iterrows():
                d = str(row["Driver"])
                t = str(row["Team"])
                driver_team_map.setdefault(d, t)

    driver_features_list: List[DriverFeatures] = []

    for drv in main_drivers:
        team = driver_team_map.get(drv, "UNKNOWN")

        # --- Pre-weekend performance features ---
        practice_score = practice_scores.get((drv, team), 0.0)

        quali_score = 0.0
        if drv in grid_from_quali:
            q_row = quali_summary[quali_summary["Driver"] == drv]
            if not q_row.empty:
                quali_score = float(q_row["quali_score"].iloc[0])

        sprint_score = sprint_scores.get(drv, 0.0)

        # Grid position: sprint result if available, else quali-based
        if use_sprint_for_grid and drv in sprint_positions:
            grid_pos = sprint_positions[drv]
        else:
            grid_pos = grid_from_quali.get(drv, len(main_drivers) + 1)

        # Practice compound / tyre features
        comp = practice_comp_feats.get((drv, team), {})
        soft_score = comp.get("soft_score", 0.0)
        medium_score = comp.get("medium_score", 0.0)
        hard_score = comp.get("hard_score", 0.0)
        avg_tyre_life_push = comp.get("avg_tyre_life_push", 0.0)
        wet_fraction = comp.get("wet_fraction", 0.0)

        # --- Previous race summary features ---
        prev_metrics = prev_summary.get(drv, {})
        last_mean_pace_rel = prev_metrics.get("mean_lap_delta_to_field", 0.0)
        last_lap_std = prev_metrics.get("lap_time_std", 0.0)
        last_positions_gained = prev_metrics.get("positions_gained", 0.0)
        last_dnf_flag = prev_metrics.get("dnf_flag", 0.0)

        # --- Persistent driver skill ---
        skill_score = skills.get(drv, 0.0)

        # --- Track features (same for all drivers this weekend) ---
        track_id = track_features.track_id
        overtaking_difficulty = track_features.overtaking_difficulty
        tyre_wear = track_features.tyre_wear

        # --- Assemble DriverFeatures ---
        df = DriverFeatures(
            year=year,
            round=rnd,
            driver=drv,
            team=team,
            grid_position=int(grid_pos),

            practice_score=float(practice_score),
            quali_score=float(quali_score),
            sprint_score=float(sprint_score),

            practice_soft_score=float(soft_score),
            practice_medium_score=float(medium_score),
            practice_hard_score=float(hard_score),
            practice_avg_tyre_life_push=float(avg_tyre_life_push),
            practice_wet_fraction=float(wet_fraction),

            last_race_mean_pace_rel=float(last_mean_pace_rel),
            last_race_lap_std=float(last_lap_std),
            last_race_positions_gained=float(last_positions_gained),
            last_race_dnf_flag=float(last_dnf_flag),

            skill_score=float(skill_score),

            track_id=track_id,
            track_overtaking_difficulty=float(overtaking_difficulty),
            track_tyre_wear=float(tyre_wear),
        )

        driver_features_list.append(df)

    return driver_features_list
