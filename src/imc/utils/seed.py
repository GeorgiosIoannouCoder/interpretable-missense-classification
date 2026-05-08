"""Deterministic seeding utilities for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (if available) RNGs.

    Parameters
    ----------
    seed : int
        Master seed used across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
