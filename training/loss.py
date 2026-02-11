# training/loss.py

"""
Loss functions for evaluating how well the model's race predictions
match actual race outcomes.

We mainly use:

    - Ignorance loss (a proper scoring rule on probabilities),
    - Mean Absolute Error (MAE) on ranks,
    - A combined loss that blends the two.

Conventions:
    - finish_distribution[driver][position] is the model's probability
      that `driver` finishes exactly in `position` (1 = winner).
    - expected_rank[driver] is E[position] from the Monte Carlo simulator.
    - actual_positions[driver] is the *true* final classified position
      from data/results.load_actual_positions.
"""

from __future__ import annotations

import math
from typing import Dict

from core_types import DriverId, FinishDist, ExpectedRank


def ignorance_loss(
    finish_dist: FinishDist,
    actual_positions: Dict[DriverId, int],
    log_base: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """
    Compute the mean ignorance score (negative log probability) of the
    model's forecasts with respect to actual outcomes.

    For each driver d:

        p_d = finish_dist[d].get(actual_positions[d], 0.0)
        IGN_d = -log_base(p_d)

    The race-level ignorance is the average over drivers for which we have
    both a forecast and an actual position.

    If p_d is zero or very small, we clamp it to `eps` to avoid log(0).

    Args:
        finish_dist:
            {driver: {position: probability}}
        actual_positions:
            {driver: actual_position (1 = winner, etc.)}
        log_base:
            Base for the logarithm; default 2.0 gives bits.
        eps:
            Minimum probability to avoid log(0).

    Returns:
        Mean ignorance across drivers (lower is better).
    """
    log_func = math.log
    denom = math.log(log_base) if log_base not in (math.e, 0) else 1.0

    ignorances = []

    for drv, actual_pos in actual_positions.items():
        dist = finish_dist.get(drv)
        if not dist:
            # No forecast for this driver; skip them
            continue

        p = dist.get(actual_pos, 0.0)
        p = max(float(p), eps)  # clamp

        # -log_base(p) = -log(p) / log(base)
        ign = -log_func(p) / denom
        ignorances.append(ign)

    if not ignorances:
        raise ValueError("ignorance_loss: No overlapping drivers between forecasts and actuals.")

    return float(sum(ignorances) / len(ignorances))


def mae_rank_loss(
    expected_rank: ExpectedRank,
    actual_positions: Dict[DriverId, int],
) -> float:
    """
    Compute the mean absolute error (MAE) of the model's expected rank
    vs actual finishing positions.

    For each driver d present in both:

        err_d = |expected_rank[d] - actual_positions[d]|

    Race-level MAE is the mean over all such drivers.

    Args:
        expected_rank:
            {driver: E[position]}
        actual_positions:
            {driver: actual_position}

    Returns:
        Mean absolute error across drivers (lower is better).
    """
    errors = []

    for drv, actual_pos in actual_positions.items():
        if drv not in expected_rank:
            continue
        exp_pos = float(expected_rank[drv])
        err = abs(exp_pos - float(actual_pos))
        errors.append(err)

    if not errors:
        raise ValueError("mae_rank_loss: No overlapping drivers between forecasts and actuals.")

    return float(sum(errors) / len(errors))


def combined_loss(
    finish_dist: FinishDist,
    expected_rank: ExpectedRank,
    actual_positions: Dict[DriverId, int],
    alpha: float = 0.7,
    log_base: float = 2.0,
) -> float:
    """
    Combine ignorance loss and MAE rank loss into a single scalar.

    combined = alpha * IGN + (1 - alpha) * MAE

    This lets you trade off between:
        - calibration of the full distribution (IGN),
        - accuracy of the mean ranking (MAE).

    Args:
        finish_dist:
            {driver: {position: probability}}
        expected_rank:
            {driver: E[position]}
        actual_positions:
            {driver: actual_position}
        alpha:
            Weight on ignorance (0..1). alpha=1.0 -> pure IGN,
            alpha=0.0 -> pure MAE.
        log_base:
            Base for the log in ignorance_loss.

    Returns:
        Scalar combined loss (lower is better).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"combined_loss: alpha must be in [0,1], got {alpha}")

    ign = ignorance_loss(finish_dist, actual_positions, log_base=log_base)
    mae = mae_rank_loss(expected_rank, actual_positions)

    return float(alpha * ign + (1.0 - alpha) * mae)
