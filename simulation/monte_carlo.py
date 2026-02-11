# simulation/monte_carlo.py

"""
Monte Carlo race simulation wrapper.

This module takes:
    - DriverFeatures for all drivers in a given race,
    - global ModelWeights,
    - TrackFeatures,
    - an RNG and number of simulations,

and repeatedly calls `simulate_race_once_laptime` to obtain many samples
of finishing order. From these samples it constructs:

    - expected_rank:  E[position] for each driver,
    - finish_distribution: P(position = k) for each driver and k.

We also optionally return a single representative lap-leaderboard trace
(from one of the MC runs) so you can visualize how a typical race
"evolves" in your model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

from core_types import (
    DriverId,
    DriverFeatures,
    ModelWeights,
    TrackFeatures,
    ExpectedRank,
    FinishDist,
    RaceSimulationResult,
)
from .lap_simulator import simulate_race_once_laptime


def _accumulate_finish_counts(
    finish_counts: Dict[DriverId, Dict[int, int]],
    finishing_order: List[DriverId],
) -> None:
    """
    Update finish_counts with the finishing_order from one simulation.

    We treat the position index in the list (0-based) as:

        position = index + 1

    Example:
        finishing_order = ["VER", "LEC", "HAM", ...]
        -> VER position 1, LEC position 2, HAM position 3, ...
    """
    for pos_idx, drv in enumerate(finishing_order):
        pos = pos_idx + 1
        finish_counts[drv][pos] += 1


def _normalize_finish_counts(
    finish_counts: Dict[DriverId, Dict[int, int]],
    n_sims: int,
) -> FinishDist:
    """
    Convert raw counts of finishes into probability distributions.

    finish_counts:
        {driver: {position: count}}

    Returns:
        finish_dist: {driver: {position: probability}}
    """
    finish_dist: FinishDist = {}
    for drv, pos_counts in finish_counts.items():
        total = float(n_sims) if n_sims > 0 else 1.0
        finish_dist[drv] = {pos: count / total for pos, count in pos_counts.items()}
    return finish_dist


def _compute_expected_ranks(
    finish_dist: FinishDist,
) -> ExpectedRank:
    """
    Compute expected finishing position for each driver from their
    finish-position distribution:

        E[position] = sum_k k * P(position = k)
    """
    expected: ExpectedRank = {}
    for drv, dist in finish_dist.items():
        exp_pos = 0.0
        for pos, p in dist.items():
            exp_pos += float(pos) * float(p)
        expected[drv] = exp_pos
    return expected


def simulate_race_mc(
    drivers: List[DriverFeatures],
    weights: ModelWeights,
    track: TrackFeatures,
    n_sims: int,
    rng: np.random.Generator,
    return_representative_trace: bool = True,
) -> RaceSimulationResult:
    """
    Run a Monte Carlo race simulation.

    Args:
        drivers:
            List of DriverFeatures for all drivers entering the race.
        weights:
            Global ModelWeights controlling feature combination and randomness.
        track:
            TrackFeatures for this circuit.
        n_sims:
            Number of Monte Carlo samples to generate.
        rng:
            NumPy random Generator for reproducibility.
        return_representative_trace:
            If True, we also keep the lap_leaderboards from one randomly
            chosen simulation (or the first one, if you prefer).

    Returns:
        RaceSimulationResult:
            expected_rank[driver], finish_distribution[driver][position],
            and optionally one lap_leaderboards trace.
    """
    if n_sims <= 0:
        raise ValueError(f"n_sims must be positive, got {n_sims}")

    # Initialize counts
    finish_counts: Dict[DriverId, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    representative_lap_trace: Optional[Dict[int, List[DriverId]]] = None
    # Optionally choose a random sim index to keep as the "representative" trace
    keep_index = rng.integers(0, n_sims) if return_representative_trace else -1

    for i in range(n_sims):
        # For reproducibility across different parameter sets, you can choose
        # to re-seed a child RNG here based on (global_seed, i). For now, we
        # just use the passed-in RNG directly.
        finishing_order, lap_leaderboards = simulate_race_once_laptime(
            drivers=drivers,
            weights=weights,
            track=track,
            rng=rng,
        )

        _accumulate_finish_counts(finish_counts, finishing_order)

        if return_representative_trace and i == keep_index:
            representative_lap_trace = lap_leaderboards

    # Build probability distributions and expected ranks
    finish_dist = _normalize_finish_counts(finish_counts, n_sims)
    expected_rank = _compute_expected_ranks(finish_dist)

    return RaceSimulationResult(
        expected_rank=expected_rank,
        finish_distribution=finish_dist,
        lap_leaderboards=representative_lap_trace,
    )

