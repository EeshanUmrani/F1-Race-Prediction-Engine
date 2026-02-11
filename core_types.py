# types.py

"""
Shared type definitions and core dataclasses for the race-sim-v2 project.

This module is intentionally small and dependency-free so it can be imported
from anywhere (data/, features/, simulation/, training/, etc.) without risk
of circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------- Basic aliases ----------

Year = int
Round = int

DriverId = str
TeamId = str
TrackId = str

# Probability distribution over finishing positions for a single driver:
#   {position (1,2,3,...) -> probability}
PositionDist = Dict[int, float]

# Distribution for all drivers:
FinishDist = Dict[DriverId, PositionDist]

ExpectedRank = Dict[DriverId, float]


# ---------- Core dataclasses ----------


@dataclass
class DriverFeatures:
    """
    Per-driver, per-race pre-event feature vector.

    This is what the feature engineering step produces and the simulator consumes.
    We include all current and planned fields, with sensible defaults so that
    older code that only sets a subset will still work.
    """

    # Identity / context
    year: Year
    round: Round
    driver: DriverId
    team: TeamId
    grid_position: int

    # --- Pre-weekend performance features (from practice/quali/sprint) ---

    # Aggregate practice race-pace score (higher = faster)
    practice_score: float = 0.0

    # Qualifying performance (higher = faster / better)
    quali_score: float = 0.0

    # Sprint performance (if sprint weekend); higher = better
    sprint_score: float = 0.0

    # Practice tyre-compound-specific scores (higher = better)
    practice_soft_score: float = 0.0
    practice_medium_score: float = 0.0
    practice_hard_score: float = 0.0

    # Average tyre life on "push laps" in practice (in laps)
    practice_avg_tyre_life_push: float = 0.0

    # Fraction of practice laps on Inter/Wet
    practice_wet_fraction: float = 0.0

    # --- In-race features from PREVIOUS race (used as inputs for next race) ---

    # Average lap-time delta to field (last race), negative = faster
    last_race_mean_pace_rel: float = 0.0

    # Lap-to-lap variability in last race (seconds)
    last_race_lap_std: float = 0.0

    # Positions gained/lost in last race (grid - finish)
    last_race_positions_gained: float = 0.0

    # 1.0 if DNF in last race, 0.0 otherwise
    last_race_dnf_flag: float = 0.0

    # --- Persistent, learned driver skill ---

    # Latent “skill” that persists across races (learned from results)
    # Convention: higher = better (tends to beat expectation)
    skill_score: float = 0.0

    # --- Track features (rolled into per-driver vector for convenience) ---

    # Optional explicit track id (e.g., 'monaco', 'interlagos')
    track_id: Optional[TrackId] = None

    # How hard it is to overtake here (0 = easy, 1 = almost impossible)
    track_overtaking_difficulty: float = 0.0

    # Relative tyre wear at this circuit (0 = very low, 1 = very high)
    track_tyre_wear: float = 0.0


@dataclass
class ModelWeights:
    """
    Global model parameters / weights.

    These control how we combine features into base performance scores and
    how much stochasticity/chaos the simulator injects.

    All fields have defaults so you can:
      - load from JSON and override only some
      - or start from defaults and let training update them.
    """

    # Linear weights on pre-weekend features
    w_practice: float = 1.0
    w_quali: float = 1.0
    w_sprint: float = 0.5
    w_grid: float = 1.0

    # Weights for previous-race & skill features
    w_last_pace_rel: float = 0.5
    w_last_positions_gained: float = 0.2
    w_skill: float = 0.5

    # Weights for compound / tyre-related practice features
    w_practice_soft: float = 0.1
    w_practice_medium: float = 0.1
    w_practice_hard: float = 0.1
    w_practice_tyre_life: float = 0.05
    w_practice_wet_fraction: float = 0.05

    # Base random noise on driver “score” used in the simulator
    noise_scale: float = 1.0

    # --- Stochastic race environment parameters (optional, learnable later) ---

    # Probability per race that a given driver gets involved in an incident / big delay
    global_incident_prob: float = 0.01

    # Approximate probability of at least one safety car period in the race
    safety_car_prob: float = 0.2

    # Additional chaos noise added to the base performance order
    chaos_noise_scale: float = 0.5


@dataclass
class TrackFeatures:
    """
    Static per-track metadata that influences both the pre-race model
    and the lap-based race simulator.
    """

    track_id: TrackId
    name: str

    # Structural properties in [0, 1]
    overtaking_difficulty: float  # 0 = easy, 1 = very hard
    tyre_wear: float              # 0 = very low, 1 = very high
    safety_car_rate: float        # approximate probability of SC in a race

    # Approximate base lap time (e.g. race pace on medium tyres in clean air)
    base_lap_time: float  # seconds

    # Number of race laps
    n_laps: int


@dataclass
class RaceSimulationResult:
    """
    Container for the outputs of a Monte Carlo race simulation.
    """

    # Expected finishing position (1 = winner) for each driver
    expected_rank: ExpectedRank

    # Finish-position probability distribution for each driver
    finish_distribution: FinishDist

    # Optional representative lap-by-lap leaderboard trace
    # (e.g. from one particular MC sample you choose to save)
    lap_leaderboards: Optional[Dict[int, List[DriverId]]] = field(default=None)
