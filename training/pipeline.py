# training/pipeline.py

"""
High-level training & prediction pipeline for a single race weekend.

This module glues together:
    - data loading (sessions, actual results),
    - feature engineering,
    - lap-based Monte Carlo simulation,
    - per-race weight updates,
    - persistent driver-skill updates,
    - race summary generation for next-race features.

Public entry points:

    run_round_predict(...)
    run_round_train(...)

Both functions operate on a SINGLE (year, round) and are expected to be
called from `main.py` in a loop over all rounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

from core_types import (
    Year,
    Round,
    DriverId,
    DriverFeatures,
    ModelWeights,
    RaceSimulationResult,
)

from data.sessions import load_sessions_for_round, load_race_laps
from data.results import load_actual_positions
from data.rounds import load_rounds_to_run
from features.build_features import build_driver_features_for_round
from simulation.monte_carlo import simulate_race_mc
from state.driver_skill import (
    load_driver_skill,
    save_driver_skill,
    update_driver_skill,
)
from state.race_summaries import (
    build_race_summary_from_laps,
    save_race_summary,
)
from state.track_features import get_track_features_for_round
from training.optimizer import update_weights_for_race
from training.loss import combined_loss


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _compute_skill_residuals(
    expected_rank: Dict[DriverId, float],
    actual_positions: Dict[DriverId, int],
) -> Dict[DriverId, float]:
    """
    Compute residuals used for updating driver skills.

    residual[drv] = expected_rank[drv] - actual_position[drv]

    Interpretation:
        residual > 0  -> driver finished BETTER than expected
                         (lower position number), so skill should go UP.
        residual < 0  -> driver finished WORSE than expected,
                         so skill should go DOWN.
    """
    residuals: Dict[DriverId, float] = {}
    for drv, actual in actual_positions.items():
        if drv not in expected_rank:
            continue
        residuals[drv] = float(expected_rank[drv]) - float(actual)
    return residuals


# -------------------------------------------------------------------
# Prediction pipeline (no training)
# -------------------------------------------------------------------


def run_round_predict(
    year: Year,
    rnd: Round,
    weights: ModelWeights,
    rng: np.random.Generator,
    n_sims: int = 5000,
    rounds: Optional[List[Tuple[Year, Round]]] = None,
    save_predictions_dir: Optional[Path] = None,
) -> RaceSimulationResult:
    """
    Run the *prediction* pipeline for a single race weekend, without
    updating any model parameters.

    Steps:
        1. Load session data for (year, round).
        2. Load the full list of rounds (if not provided).
        3. Build DriverFeatures (including previous-race & skill features).
        4. Load TrackFeatures.
        5. Run Monte Carlo race simulation.
        6. Optionally save the prediction to disk.

    Args:
        year, rnd:
            Race weekend identifier.
        weights:
            Current ModelWeights (not modified).
        rng:
            NumPy RNG for Monte Carlo sampling.
        n_sims:
            Number of Monte Carlo samples for the simulation.
        rounds:
            Sorted list of all (year, round) to run. If None, it will be
            loaded from data/rounds.load_rounds_to_run().
        save_predictions_dir:
            If provided, a JSON file with the prediction will be written
            here as `<year>_<round>.json`.

    Returns:
        RaceSimulationResult with expected_rank, finish_distribution,
        and one representative lap-leaderboard trace.
    """
    # 1. Load sessions for this weekend
    sessions_by_type = load_sessions_for_round(year, rnd)

    # 2. Full rounds list (for previous-race lookup inside build_features)
    if rounds is None:
        rounds = load_rounds_to_run()

    # 3. Features
    driver_features: List[DriverFeatures] = build_driver_features_for_round(
        year=year,
        rnd=rnd,
        rounds=rounds,
        sessions_by_type=sessions_by_type,
    )

    if not driver_features:
        raise ValueError(f"run_round_predict: No driver features for {year} round {rnd}")

    # 4. TrackFeatures
    track_features = get_track_features_for_round(year, rnd)

    # 5. Monte Carlo simulation
    sim_result = simulate_race_mc(
        drivers=driver_features,
        weights=weights,
        track=track_features,
        n_sims=n_sims,
        rng=rng,
        return_representative_trace=True,
    )

    # 6. Optional saving to disk
    if save_predictions_dir is not None:
        _save_prediction_to_disk(
            year=year,
            rnd=rnd,
            result=sim_result,
            out_dir=save_predictions_dir,
        )

    return sim_result


def _save_prediction_to_disk(
    year: Year,
    rnd: Round,
    result: RaceSimulationResult,
    out_dir: Path,
) -> None:
    """
    Serialize a RaceSimulationResult to JSON for later inspection.

    This is optional; if you don't care about saving predictions, you
    can ignore / not call this helper.
    """
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{year}_{rnd:02d}.json"

    payload = {
        "year": year,
        "round": rnd,
        "expected_rank": result.expected_rank,
        "finish_distribution": {
            drv: {int(pos): float(p) for pos, p in dist.items()}
            for drv, dist in result.finish_distribution.items()
        },
        "lap_leaderboards": {
            int(lap): list(order)
            for lap, order in (result.lap_leaderboards or {}).items()
        },
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -------------------------------------------------------------------
# Training pipeline (updates weights, skills, and race summaries)
# -------------------------------------------------------------------


def run_round_train(
    year: Year,
    rnd: Round,
    weights: ModelWeights,
    rng: np.random.Generator,
    n_sims: int = 3000,
    rounds: Optional[List[Tuple[Year, Round]]] = None,
    alpha_loss: float = 0.7,
    alpha_skill: float = 0.1,
    verbose: bool = False,
) -> Tuple[ModelWeights, float]:
    """
    Run the *training* pipeline for a single race weekend.

    Steps:
        1. Load session data for (year, round).
        2. Load actual finishing positions (ground truth).
        3. Build DriverFeatures (including previous-race & skill features).
        4. Load TrackFeatures.
        5. Optimize ModelWeights for this race via coordinate search.
        6. Run a final simulation with the updated weights and compute loss.
        7. Update persistent driver skills using residuals.
        8. Build and save race summary from actual laps for use by the
           *next* race's features.

    Args:
        year, rnd:
            Race weekend identifier.
        weights:
            Current ModelWeights (a new, possibly updated copy is returned).
        rng:
            NumPy RNG for Monte Carlo sampling and optimization.
        n_sims:
            Number of MC samples per evaluation inside the optimizer.
        rounds:
            Sorted list of all (year, round). If None, loaded automatically.
        alpha_loss:
            Weighting between ignorance and MAE in the combined loss
            used by the optimizer.
        alpha_skill:
            Learning rate for driver skill updates (EMA coefficient).
        verbose:
            If True, prints optimization progress.

    Returns:
        (updated_weights, final_loss)
            updated_weights: possibly improved ModelWeights.
            final_loss: combined loss on this race using updated weights.
    """
    # 1. Load sessions for this weekend
    sessions_by_type = load_sessions_for_round(year, rnd)

    # 2. Load actual results
    actual_positions = load_actual_positions(year, rnd)

    # 3. Full rounds list (for previous-race lookup)
    if rounds is None:
        rounds = load_rounds_to_run()

    # 4. Build DriverFeatures
    driver_features: List[DriverFeatures] = build_driver_features_for_round(
        year=year,
        rnd=rnd,
        rounds=rounds,
        sessions_by_type=sessions_by_type,
    )
    if not driver_features:
        raise ValueError(f"run_round_train: No driver features for {year} round {rnd}")

    # 5. TrackFeatures
    track_features = get_track_features_for_round(year, rnd)

    # 6. Optimize weights for this race
    updated_weights = update_weights_for_race(
        weights=weights,
        drivers=driver_features,
        track=track_features,
        actual_positions=actual_positions,
        n_sims=n_sims,
        rng=rng,
        alpha_loss=alpha_loss,
        verbose=verbose,
    )

    # 7. Run a final simulation with updated weights for:
    #       - final loss reporting
    #       - skill residuals
    final_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
    sim_result = simulate_race_mc(
        drivers=driver_features,
        weights=updated_weights,
        track=track_features,
        n_sims=n_sims,
        rng=final_rng,
        return_representative_trace=False,
    )

    final_loss = combined_loss(
        finish_dist=sim_result.finish_distribution,
        expected_rank=sim_result.expected_rank,
        actual_positions=actual_positions,
        alpha=alpha_loss,
    )

    if verbose:
        print(f"[TRAIN] {year} R{rnd:02d}: final combined loss = {final_loss:.4f}")

    # 8. Update persistent driver skills
    skills = load_driver_skill()
    residuals = _compute_skill_residuals(
        expected_rank=sim_result.expected_rank,
        actual_positions=actual_positions,
    )
    update_driver_skill(skills, residuals, alpha=alpha_skill)
    save_driver_skill(skills)

    # 9. Build & save race summary from actual laps
    race_laps = load_race_laps(year, rnd)
    summary = build_race_summary_from_laps(race_laps)
    save_race_summary(year, rnd, summary)

    return updated_weights, final_loss
