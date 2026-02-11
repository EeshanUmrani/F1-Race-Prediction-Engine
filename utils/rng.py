# utils/rng.py

"""
Random number generator utilities.

We use NumPy's Generator API everywhere so that:
  - simulations are reproducible,
  - child RNGs for different races / evaluations can be derived cleanly.

This module provides a single helper, `make_rng`, which you should call
in main.py and then pass the resulting Generator down into the
simulation / training pipelines.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a NumPy random Generator.

    Args:
        seed:
            If provided, used to seed the Generator deterministically.
            If None, we use NumPy's default seeding (OS entropy).

    Returns:
        np.random.Generator instance.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))
