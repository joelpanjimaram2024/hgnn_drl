"""Reproducibility utilities.

Sets seeds for Python, NumPy, PyTorch, and (optionally) CUDA / cuDNN,
then verifies that training is deterministic on the given device.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, *, device: str = "cpu", deterministic_cudnn: bool = True) -> None:
    """Set all RNG seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Base seed for all random number generators.
    device : str, optional
        Torch device string (e.g. "cpu", "cuda").
    deterministic_cudnn : bool, optional
        If True, enable cuDNN deterministic algorithms where supported.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Guarantee identical CPU dispatch when the helper derives it.
    if device in ("cpu", "cpu:", ""):
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Partial reproducibility for non-deterministic ops is not possible;
    # callers must treat seeds as a hint, not a hard guarantee.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def reset_seed(seed: int) -> None:
    """Convenience wrapper that simply re-applies the seed set."""
    set_seed(seed)


def verify_dropout_seed(seed: int, n_channels: int = 2, n_tasks: int = 4) -> dict:
    """Run a tiny forward pass twice under the same seed to check dropout
    determinism on the current device.  Returns a dict of booleans."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(seed)
    x = torch.randn(n_tasks, n_channels, device=device)
    m = torch.nn.Dropout(0.2)
    a = m(x)
    b = m(x)
    return {"dropout_det": torch.equal(a, b), "device": device}


def train_seed_sequence(base_seed: int, n_runs: int) -> list[int]:
    """Return a list of reproducible seeds for training runs."""
    rng = np.random.default_rng(base_seed)
    return [int(rng.integers(0, 1 << 31)) for _ in range(n_runs)]
