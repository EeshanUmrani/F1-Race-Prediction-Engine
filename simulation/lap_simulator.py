# simulation/lap_simulator.py

"""
Lap-based single-race simulator.

This module simulates ONE Monte Carlo realization of a race, lap by lap,
given:

    - pre-race DriverFeatures for all drivers,
    - global ModelWeights,
    - TrackFeatures for the circuit,
    - an RNG.

It returns:
    - finishing_order: list[DriverId] sorted by final classification
      (all finishers first, then DNFs ordered by laps completed / time),
    - lap_leaderboards: dict[lap_number -> list[DriverId]] giving the
      running order at the end of each lap.

Notes / design choices:
    - This is intentionally *simple but structured* so you can refine it later.
    - Grid position affects base pace via the linear model (through w_grid).
    - Lap times come from a base time per driver with:
        * linear combination of features (practice/quali/sprint/etc)
        * track.base_lap_time
        * tyre wear factor (via track.tyre_wear)
        * driver-level chaos (ModelWeights.chaos_noise_scale)
        * lap-to-lap noise (ModelWeights.noise_scale)
    - DNFs are modeled with a small per-lap incident probability
      (ModelWeights.global_incident_prob).
    - Safety car is not explicitly modeled yet; we can add that later by
      manipulating gaps / lap times in specific laps.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from core_types import (
    DriverId,
    DriverFeatures,
    ModelWeights,
    TrackFeatures,
)


# -------------------------------------------------------------------
# Helpers to turn DriverFeatures into a base pace (lap time) model
# -------------------------------------------------------------------


def _compute_driver_score(df: DriverFeatures, w: ModelWeights) -> float:
    """
    Compute a scalar 'pace score' for a driver using a linear model.

    Higher score -> faster driver (shorter lap time).
    """
    # Core pre-weekend features
    score = 0.0
    score += w.w_practice * df.practice_score
    score += w.w_quali * df.quali_score
    score += w.w_sprint * df.sprint_score
    score += w.w_grid * (-float(df.grid_position))  # lower grid position (1) -> higher score

    # Previous race & skill
    score += w.w_last_pace_rel * (-df.last_race_mean_pace_rel)  # negative delta (faster) -> boost
    score += w.w_last_positions_gained * df.last_race_positions_gained
    score += w.w_skill * df.skill_score

    # Tyre-related practice features
    score += w.w_practice_soft * df.practice_soft_score
    score += w.w_practice_medium * df.practice_medium_score
    score += w.w_practice_hard * df.practice_hard_score
    score += w.w_practice_tyre_life * df.practice_avg_tyre_life_push
    score += w.w_practice_wet_fraction * df.practice_wet_fraction

    return float(score)


def _score_to_base_lap_time(
    base_track_lap: float,
    score: float,
    pace_scale: float = 0.15,
) -> float:
    """
    Map a driver's scalar score to a base lap time in seconds.

    A simple linear mapping:

        lap_time = base_track_lap - pace_scale * score

    So a +1 difference in score reduces lap time by `pace_scale` seconds.
    """
    return float(base_track_lap - pace_scale * score)


# -------------------------------------------------------------------
# Main simulation
# -------------------------------------------------------------------


def simulate_race_once_laptime(
    drivers: List[DriverFeatures],
    weights: ModelWeights,
    track: TrackFeatures,
    rng: np.random.Generator,
) -> Tuple[List[DriverId], Dict[int, List[DriverId]]]:
    """
    Simulate one realization of a race via lap times.

    Args:
        drivers:
            List of DriverFeatures for the weekend. All drivers in this list
            are assumed to start the race.
        weights:
            Global ModelWeights controlling how features combine and how much
            stochasticity is present.
        track:
            TrackFeatures for this circuit.
        rng:
            NumPy Generator for randomness.

    Returns:
        finishing_order:
            List of driver IDs sorted by final classification.
        lap_leaderboards:
            Mapping from lap_number -> list[driver_id] representing the order
            at the end of each lap.
    """
    if not drivers:
        return [], {}

    n_laps = track.n_laps
    if n_laps <= 0:
        raise ValueError(f"Track {track.track_id} has non-positive n_laps={n_laps}")

    # Pre-compute base scores and base lap times
    base_scores: Dict[DriverId, float] = {}
    base_lap_times: Dict[DriverId, float] = {}

    for df in drivers:
        drv = df.driver
        s = _compute_driver_score(df, weights)
        base_scores[drv] = s

        # Add a bit of driver-level "chaos" to their base pace
        chaos = rng.normal(0.0, weights.chaos_noise_scale)
        base_lap_times[drv] = max(
            1.0,  # guard against nonsense
            _score_to_base_lap_time(track.base_lap_time, s) + chaos,
        )

    # Tyre degradation factor:
    #   We'll apply a simple multiplicative increase over the race, scaled by track.tyre_wear:
    #       lap_deg_factor(lap) = 1 + (track.tyre_wear * deg_scale) * (lap / n_laps)
    #   so the last lap is at most ~deg_scale * track.tyre_wear slower than the first.
    deg_scale = 0.4  # at tyre_wear=1.0, last lap can be ~1.4x slower than first
    lap_indices = np.arange(1, n_laps + 1, dtype=float)

    # Lap-to-lap noise
    lap_noise_std = max(0.0, float(weights.noise_scale))

    # Incident probability: per-lap chance of DNF for each active driver.
    p_incident = max(0.0, min(1.0, float(weights.global_incident_prob)))

    # State: cumulative time and DNF flag per driver
    cumulative_time: Dict[DriverId, float] = {df.driver: 0.0 for df in drivers}
    is_dnf: Dict[DriverId, bool] = {df.driver: False for df in drivers}
    laps_completed: Dict[DriverId, int] = {df.driver: 0 for df in drivers}

    lap_leaderboards: Dict[int, List[DriverId]] = {}

    for lap in lap_indices.astype(int):
        # Compute tyre-degradation multiplier for this lap
        tyre_factor = 1.0 + (track.tyre_wear * deg_scale) * (lap / float(n_laps))

        for df in drivers:
            drv = df.driver
            if is_dnf[drv]:
                continue  # no further laps for this driver

            # Possible incident / DNF
            if p_incident > 0.0 and rng.random() < p_incident:
                # Model a DNF: add a modest time chunk for this lap then mark as DNF
                # (so they appear behind finishers but still with a sensible time)
                incident_penalty = rng.uniform(20.0, 60.0)  # seconds lost before retiring
                base = base_lap_times[drv] * tyre_factor
                noise = rng.normal(0.0, lap_noise_std)
                lap_time = max(0.5, base + noise + incident_penalty)

                cumulative_time[drv] += lap_time
                laps_completed[drv] = lap
                is_dnf[drv] = True
                continue

            # Normal lap
            base = base_lap_times[drv] * tyre_factor
            noise = rng.normal(0.0, lap_noise_std)
            lap_time = max(0.5, base + noise)

            cumulative_time[drv] += lap_time
            laps_completed[drv] = lap

        # Leaderboard at end of this lap: sort by (DNF flag, cumulative_time)
        # All non-DNF cars, ordered by time; DNFs included as well (worse).
        ordered = sorted(
            cumulative_time.keys(),
            key=lambda d: (is_dnf[d], cumulative_time[d]),
        )
        lap_leaderboards[int(lap)] = ordered.copy()

    # Final classification
    # First by DNF (False < True), then by laps_completed (more laps is better),
    # then by cumulative_time.
    finishing_order = sorted(
        cumulative_time.keys(),
        key=lambda d: (
            is_dnf[d],
            -laps_completed[d],    # more laps first
            cumulative_time[d],
        ),
    )

    return finishing_order, lap_leaderboards
