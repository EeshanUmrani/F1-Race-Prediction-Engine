# training/optimizer.py

"""
Simple optimizer for per-race weight updates.

This module provides a hill-climbing / coordinate search style optimizer
that adjusts ModelWeights to reduce the loss on a SINGLE race.

High-level idea:
    - Start from current weights.
    - For a small number of iterations:
        * For each tunable parameter:
            - Try +step change, evaluate loss via Monte Carlo simulation.
            - Try -step change, evaluate loss.
            - If either improves loss, keep the best move.
        * Optionally shrink step size after each full sweep.

This is intentionally simple and local: it's not trying to find a global
optimum, just nudging weights in a direction that improves performance
for the current race, which is good enough for an online / incremental
training scheme across many races.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

from core_types import (
    DriverFeatures,
    ModelWeights,
    TrackFeatures,
    Year,
    Round,
)
from simulation.monte_carlo import simulate_race_mc
from .loss import combined_loss


# -------------------------------------------------------------------
# Internal evaluation helper
# -------------------------------------------------------------------


def _evaluate_weights_on_race(
    weights: ModelWeights,
    drivers: List[DriverFeatures],
    track: TrackFeatures,
    actual_positions: Dict[str, int],
    n_sims: int,
    global_rng: np.random.Generator,
    alpha_loss: float = 0.7,
) -> float:
    """
    Evaluate a given set of weights on a single race.

    We run a Monte Carlo simulation with `n_sims` samples and compute
    the combined loss (ignorance + MAE) against the actual results.

    We use a child RNG for the simulation so that different calls are
    decorrelated but still reproducible given the parent RNG.
    """
    # Derive a child RNG from the global one for this evaluation
    seed = int(global_rng.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)

    sim_result = simulate_race_mc(
        drivers=drivers,
        weights=weights,
        track=track,
        n_sims=n_sims,
        rng=rng,
        return_representative_trace=False,
    )

    loss_val = combined_loss(
        finish_dist=sim_result.finish_distribution,
        expected_rank=sim_result.expected_rank,
        actual_positions=actual_positions,
        alpha=alpha_loss,
    )
    return float(loss_val)


# -------------------------------------------------------------------
# Public optimizer API
# -------------------------------------------------------------------


def update_weights_for_race(
    weights: ModelWeights,
    drivers: List[DriverFeatures],
    track: TrackFeatures,
    actual_positions: Dict[str, int],
    n_sims: int,
    rng: np.random.Generator,
    alpha_loss: float = 0.7,
    initial_step: float = 0.25,
    min_step: float = 0.01,
    step_decay: float = 0.5,
    max_sweeps: int = 3,
    tunable_params: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> ModelWeights:
    """
    Perform a small coordinate-descent style update of ModelWeights
    based on a single race.

    Args:
        weights:
            Current ModelWeights (will not be modified; a copy is returned).
        drivers:
            DriverFeatures for all drivers in the race.
        track:
            TrackFeatures for this circuit.
        actual_positions:
            Mapping driver -> actual finishing position.
        n_sims:
            Number of Monte Carlo samples to use per evaluation.
        rng:
            NumPy random Generator for reproducibility.
        alpha_loss:
            Weight for ignorance vs MAE in the combined loss.
        initial_step:
            Initial magnitude of parameter perturbations.
        min_step:
            Minimum step size; once step < min_step, optimization stops.
        step_decay:
            Factor by which the step is shrunk after each full parameter sweep.
        max_sweeps:
            Maximum number of sweeps over all tunable parameters.
        tunable_params:
            Iterable of attribute names on ModelWeights to tune.
            If None, we use a sensible default set.
        verbose:
            If True, prints loss improvements and parameter changes.

    Returns:
        New ModelWeights with (hopefully) lower loss than the original.
        If no beneficial move is found, the original weights are returned.
    """
    if tunable_params is None:
        # Default: focus on linear feature weights and noise/chaos scales
        tunable_params = [
            "w_practice",
            "w_quali",
            "w_sprint",
            "w_grid",
            "w_last_pace_rel",
            "w_last_positions_gained",
            "w_skill",
            "w_practice_soft",
            "w_practice_medium",
            "w_practice_hard",
            "w_practice_tyre_life",
            "w_practice_wet_fraction",
            "noise_scale",
            "chaos_noise_scale",
            # You can add "global_incident_prob", "safety_car_prob" later if desired
        ]

    tunable_params = list(tunable_params)

    # Parameter-specific bounds (min, max) to keep things sane.
    # Values outside these ranges are clamped.
    param_bounds: Dict[str, Tuple[float, float]] = {
        "w_practice": (-5.0, 5.0),
        "w_quali": (-5.0, 5.0),
        "w_sprint": (-5.0, 5.0),
        "w_grid": (-5.0, 5.0),
        "w_last_pace_rel": (-5.0, 5.0),
        "w_last_positions_gained": (-5.0, 5.0),
        "w_skill": (-5.0, 5.0),
        "w_practice_soft": (-5.0, 5.0),
        "w_practice_medium": (-5.0, 5.0),
        "w_practice_hard": (-5.0, 5.0),
        "w_practice_tyre_life": (-5.0, 5.0),
        "w_practice_wet_fraction": (-5.0, 5.0),
        "noise_scale": (0.0, 5.0),
        "chaos_noise_scale": (0.0, 5.0),
        "global_incident_prob": (0.0, 0.5),
        "safety_car_prob": (0.0, 0.9),
    }

    # Clone the weights to avoid mutating the input
    current = copy.deepcopy(weights)
    best_loss = _evaluate_weights_on_race(
        current,
        drivers=drivers,
        track=track,
        actual_positions=actual_positions,
        n_sims=n_sims,
        global_rng=rng,
        alpha_loss=alpha_loss,
    )

    if verbose:
        print(f"Initial loss: {best_loss:.4f}")

    step = float(initial_step)

    for sweep in range(max_sweeps):
        improved_in_sweep = False

        if verbose:
            print(f"\nSweep {sweep + 1}/{max_sweeps}, step={step:.4f}")

        for param in tunable_params:
            if not hasattr(current, param):
                continue

            base_val = float(getattr(current, param))

            # Try +step
            plus_weights = copy.deepcopy(current)
            plus_val = base_val + step
            if param in param_bounds:
                lo, hi = param_bounds[param]
                plus_val = max(lo, min(hi, plus_val))
            setattr(plus_weights, param, plus_val)

            plus_loss = _evaluate_weights_on_race(
                plus_weights,
                drivers=drivers,
                track=track,
                actual_positions=actual_positions,
                n_sims=n_sims,
                global_rng=rng,
                alpha_loss=alpha_loss,
            )

            # Try -step
            minus_weights = copy.deepcopy(current)
            minus_val = base_val - step
            if param in param_bounds:
                lo, hi = param_bounds[param]
                minus_val = max(lo, min(hi, minus_val))
            setattr(minus_weights, param, minus_val)

            minus_loss = _evaluate_weights_on_race(
                minus_weights,
                drivers=drivers,
                track=track,
                actual_positions=actual_positions,
                n_sims=n_sims,
                global_rng=rng,
                alpha_loss=alpha_loss,
            )

            # Decide if any move is good
            candidate_loss = best_loss
            candidate_weights = current
            candidate_val = base_val
            direction = "none"

            if plus_loss < candidate_loss:
                candidate_loss = plus_loss
                candidate_weights = plus_weights
                candidate_val = plus_val
                direction = "+"

            if minus_loss < candidate_loss:
                candidate_loss = minus_loss
                candidate_weights = minus_weights
                candidate_val = minus_val
                direction = "-"

            if candidate_loss < best_loss - 1e-6:  # small tolerance
                if verbose:
                    print(
                        f"  {param}: {base_val:.3f} -> {candidate_val:.3f} "
                        f"({direction}), loss {best_loss:.4f} -> {candidate_loss:.4f}"
                    )
                current = candidate_weights
                best_loss = candidate_loss
                improved_in_sweep = True
            else:
                if verbose:
                    # Show that this parameter had no beneficial move
                    print(
                        f"  {param}: no improvement at step={step:.3f} "
                        f"(best_loss={best_loss:.4f})"
                    )

        # After trying all params:
        if not improved_in_sweep:
            # No improvement: shrink step size and possibly exit
            step *= step_decay
            if verbose:
                print(f"No improvement in sweep; shrinking step to {step:.4f}")
            if step < min_step:
                if verbose:
                    print("Step size below min_step; stopping optimization.")
                break

    if verbose:
        print("\nOptimization complete.")
        print("Final weights:", asdict(current))
        print(f"Final loss: {best_loss:.4f}")

    return current
