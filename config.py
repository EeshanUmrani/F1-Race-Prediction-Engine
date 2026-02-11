# config.py

"""
Global configuration for the race-sim-v2 project.

This module centralizes:
  - filesystem paths,
  - default Monte Carlo settings,
  - run mode (train vs predict),
  - misc knobs you might want to tweak from a single place.

Other modules *can* import from here, but they don't have to – most of
them will still work with their own sensible defaults if you prefer to
wire things manually in main.py.
"""

from __future__ import annotations

from pathlib import Path


# -------------------------------------------------------------------
# Core paths
# -------------------------------------------------------------------

# Root of the project (directory containing main.py, config.py, etc.)
PROJECT_ROOT: Path = Path("C:/Coding/Python/NAP Project 2/race-sim-v2")

# Data directory (sessions_index.json, raw parquet files, etc.)
DATA_DIR: Path = PROJECT_ROOT / "data"

# State directory (weights, driver skills, race summaries, track meta)
STATE_DIR: Path = PROJECT_ROOT / "state"

# Directory to save predictions (optional)
PREDICTIONS_DIR: Path = PROJECT_ROOT / "predictions"

# Directory for logs (if you want to write logs to disk)
LOGS_DIR: Path = PROJECT_ROOT / "logs"


# -------------------------------------------------------------------
# Run mode & general settings
# -------------------------------------------------------------------

# Either "train" or "predict".
# main.py can read this to decide which pipeline to run.
RUN_MODE: str = "train"  # or "predict"

# Global RNG seed for reproducibility
RANDOM_SEED: int = 42


# -------------------------------------------------------------------
# Monte Carlo settings
# -------------------------------------------------------------------

# Default number of Monte Carlo samples used when *predicting* a race.
N_SIMS_PREDICT: int = 20000

# Default number of Monte Carlo samples used per evaluation when *training*.
N_SIMS_TRAIN: int = 1000

# How strongly to weight ignorance vs MAE in the combined loss.
# alpha = 1.0  -> pure ignorance
# alpha = 0.0  -> pure MAE
ALPHA_LOSS: float = 0.7

# Learning rate for driver skill updates (EMA coefficient).
ALPHA_SKILL: float = 0.1


# -------------------------------------------------------------------
# Optimizer defaults (can be overridden in training.pipeline.run_round_train)
# -------------------------------------------------------------------

INITIAL_STEP: float = 0.25
MIN_STEP: float = 0.01
STEP_DECAY: float = 0.5
MAX_SWEEPS: int = 3
