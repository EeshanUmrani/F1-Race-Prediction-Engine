# main.py

"""
Entry point for the race-sim-v2 project.

Typical usage:

    # Train across all available rounds
    python main.py --mode train

    # Run predictions (no training) across all rounds
    python main.py --mode predict

You can also restrict to a subset of rounds with:
    python main.py --mode predict --year 2025 --round 24

This script wires together:
    - config (paths, seeds, defaults),
    - data.rounds.load_rounds_to_run,
    - state.weights_store (load/save ModelWeights),
    - training.pipeline (run_round_train / run_round_predict),
    - utils.rng.make_rng (reproducible RNG).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import (
    PROJECT_ROOT,
    PREDICTIONS_DIR,
    RUN_MODE,
    RANDOM_SEED,
    N_SIMS_PREDICT,
    N_SIMS_TRAIN,
    ALPHA_LOSS,
    ALPHA_SKILL,
)
from data.rounds import load_rounds_to_run
from state.weights_store import load_weights, save_weights
from training.pipeline import run_round_train, run_round_predict
from utils.rng import make_rng


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 race simulation v2")

    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default=RUN_MODE,
        help=f"Run mode (default from config.py: {RUN_MODE!r})",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="If provided, restrict to this single year.",
    )

    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="If provided (with --year), restrict to this single round.",
    )

    parser.add_argument(
        "--n-sims-train",
        type=int,
        default=N_SIMS_TRAIN,
        help=f"Number of Monte Carlo samples per evaluation in TRAIN mode "
             f"(default: {N_SIMS_TRAIN})",
    )

    parser.add_argument(
        "--n-sims-predict",
        type=int,
        default=N_SIMS_PREDICT,
        help=f"Number of Monte Carlo samples in PREDICT mode "
             f"(default: {N_SIMS_PREDICT})",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for RNG (default from config.py: {RANDOM_SEED})",
    )

    parser.add_argument(
        "--alpha-loss",
        type=float,
        default=ALPHA_LOSS,
        help=f"Weight for ignorance vs MAE in combined loss (default: {ALPHA_LOSS})",
    )

    parser.add_argument(
        "--alpha-skill",
        type=float,
        default=ALPHA_SKILL,
        help=f"Learning rate for driver skill updates (default: {ALPHA_SKILL})",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra information during training.",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help=f"Save predictions to {PREDICTIONS_DIR.relative_to(PROJECT_ROOT)}/*.json "
             f"in PREDICT mode.",
    )

    return parser.parse_args()


def _filter_rounds(
    rounds: List[Tuple[int, int]],
    year: int | None,
    rnd: int | None,
) -> List[Tuple[int, int]]:
    """
    Restrict the list of (year, round) pairs based on CLI args, if provided.
    """
    if year is None:
        return rounds

    filtered = [r for r in rounds if r[0] == year]
    if rnd is not None:
        filtered = [r for r in filtered if r[1] == rnd]

    return filtered


def main() -> None:
    args = _parse_args()
    
    # If no CLI arguments are given at all, default to train + verbose
    if len(sys.argv) == 1:
        args.mode = "train"
        args.verbose = True


    mode = args.mode
    seed = args.seed
    n_sims_train = args.n_sims_train
    n_sims_predict = args.n_sims_predict
    alpha_loss = args.alpha_loss
    alpha_skill = args.alpha_skill
    verbose = args.verbose

    # Load list of all rounds from sessions_index.json
    all_rounds = load_rounds_to_run()
    rounds = _filter_rounds(all_rounds, args.year, args.round)
    
    if not rounds and mode == "predict" and args.year is not None and args.round is not None:
        print(
            f"Requested ({args.year}, {args.round}) is not in rounds_to_run "
            f"(probably missing race session); running it anyway for prediction."
            )
        rounds = [(args.year, args.round)]

    if not rounds:
        raise SystemExit("No rounds to run. Check your --year/--round filters and data.")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Mode: {mode}")
    print(f"Rounds to run: {rounds}")
    print(f"Random seed: {seed}")

    # Global RNG
    rng = make_rng(seed)

    # Load global model weights (from JSON or defaults)
    weights = load_weights()
    print("Loaded model weights.")

    if mode == "train":
        for (year, rnd) in rounds:
            print(f"\n=== TRAIN: {year} Round {rnd:02d} ===")
            weights, final_loss = run_round_train(
                year=year,
                rnd=rnd,
                weights=weights,
                rng=rng,
                n_sims=n_sims_train,
                rounds=all_rounds,
                alpha_loss=alpha_loss,
                alpha_skill=alpha_skill,
                verbose=verbose,
            )
            print(f"[TRAIN] {year} R{rnd:02d} final loss: {final_loss:.4f}")
            save_weights(weights)
            print("Weights saved.")

    elif mode == "predict":
        predictions_dir: Path | None = PREDICTIONS_DIR if args.save_predictions else None
        if predictions_dir is not None:
            print(f"Predictions will be saved to: {predictions_dir}")

        for (year, rnd) in rounds:
            print(f"\n=== PREDICT: {year} Round {rnd:02d} ===")
            result = run_round_predict(
                year=year,
                rnd=rnd,
                weights=weights,
                rng=rng,
                n_sims=n_sims_predict,
                rounds=all_rounds,
                save_predictions_dir=predictions_dir,
            )

            # Print a simple summary to stdout
            ranked = sorted(result.expected_rank.items(), key=lambda kv: kv[1])
            print("  Expected finishing order (driver, E[position]):")
            for drv, exp_pos in ranked:
                print(f"    {drv:>3}  {exp_pos:5.2f}")

    else:
        raise SystemExit(f"Unknown mode: {mode!r}")


if __name__ == "__main__":
    main()
