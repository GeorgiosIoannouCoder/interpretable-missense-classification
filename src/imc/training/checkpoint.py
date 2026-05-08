"""SSH-drop-safe checkpoint helpers for the ESM-2 head training loop.

A checkpoint is a single ``torch.save``-d ``.pt`` file containing
``model``, ``optimizer``, ``scheduler``, ``scaler`` (if any), ``epoch``,
``global_step``, ``best_val_metric``, and the various RNG states. We
maintain two files:

- ``last.pt`` — the most recent checkpoint (atomic-rename-written).
- ``best.pt`` — the checkpoint with the best validation metric so far.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class CheckpointState:
    """In-memory checkpoint payload."""

    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None
    epoch: int
    global_step: int
    best_val_metric: float
    rng_python: tuple
    rng_numpy: dict[str, Any]
    rng_torch: torch.Tensor
    rng_torch_cuda: list[torch.Tensor] | None


def gather_rng_state() -> tuple[tuple, dict[str, Any], torch.Tensor, list[torch.Tensor] | None]:
    """Snapshot Python / NumPy / Torch / CUDA RNG states."""
    py = random.getstate()
    npr = np.random.get_state()  # actually a tuple of length 5; np.random.set_state accepts it back
    npr_dict = {"state": npr}
    th = torch.get_rng_state()
    th_cuda = (
        [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else None
    )
    return py, npr_dict, th, th_cuda


def restore_rng_state(
    py: tuple,
    npr: dict[str, Any],
    th: torch.Tensor,
    th_cuda: list[torch.Tensor] | None,
) -> None:
    """Restore RNG states snapshotted by :func:`gather_rng_state`."""
    random.setstate(py)
    np.random.set_state(npr["state"])
    torch.set_rng_state(th)
    if th_cuda and torch.cuda.is_available():
        for i, s in enumerate(th_cuda):
            torch.cuda.set_rng_state(s, device=i)


def save_checkpoint(path: str | Path, state: CheckpointState) -> None:
    """Atomically write ``state`` to ``path`` (write tmp, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": state.model_state,
        "optimizer": state.optimizer_state,
        "scheduler": state.scheduler_state,
        "scaler": state.scaler_state,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_val_metric": state.best_val_metric,
        "rng_python": state.rng_python,
        "rng_numpy": state.rng_numpy,
        "rng_torch": state.rng_torch,
        "rng_torch_cuda": state.rng_torch_cuda,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path) -> CheckpointState:
    """Load a checkpoint produced by :func:`save_checkpoint`."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return CheckpointState(
        model_state=payload["model"],
        optimizer_state=payload["optimizer"],
        scheduler_state=payload.get("scheduler"),
        scaler_state=payload.get("scaler"),
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        best_val_metric=float(payload["best_val_metric"]),
        rng_python=payload["rng_python"],
        rng_numpy=payload["rng_numpy"],
        rng_torch=payload["rng_torch"],
        rng_torch_cuda=payload.get("rng_torch_cuda"),
    )
