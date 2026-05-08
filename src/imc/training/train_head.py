"""Training loop for the ESM-2 MLP head over cached residue embeddings.

Single-GPU and multi-GPU (via Hugging Face ``accelerate``) work without
code changes. Checkpoints are SSH-drop safe: we save ``last.pt`` every
``ckpt_steps`` and after each epoch, plus ``best.pt`` whenever validation
AUPRC improves. Pass ``--resume auto`` (the default) to pick up where a
killed run left off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from imc.models.head import ESM2HeadMLP
from imc.training.checkpoint import (
    CheckpointState,
    gather_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from imc.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass
class TrainHeadConfig:
    """Hyperparameters for the ESM-2 MLP head training loop.

    Attributes
    ----------
    in_dim : int
        Input embedding dimensionality (1280 for ESM-2 650M).
    hidden : int
        MLP hidden-layer width.
    dropout : float
        Dropout probability.
    norm : str
        ``"layer"`` or ``"batch"``.
    lr : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Per-rank batch size.
    sampler : str
        ``"class_weight"`` (BCE pos_weight) or ``"balanced"``
        (``WeightedRandomSampler``).
    early_stop_patience : int
        Stop if val AUPRC fails to improve for this many epochs.
    ckpt_steps : int
        Save ``last.pt`` every this many optimizer steps.
    seed : int
        Master random seed.
    """

    in_dim: int = 1280
    hidden: int = 256
    dropout: float = 0.2
    norm: str = "layer"
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 25
    batch_size: int = 512
    sampler: str = "class_weight"
    early_stop_patience: int = 5
    ckpt_steps: int = 500
    seed: int = 42


@torch.inference_mode()
def _evaluate(model: ESM2HeadMLP, loader: DataLoader, device: str) -> tuple[float, float]:
    """Compute AUPRC and AUROC on a single-process evaluation loader."""
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        scores = torch.sigmoid(logits).float().detach().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(y.detach().cpu().numpy())
    s = np.concatenate(all_scores)
    y = np.concatenate(all_labels)
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sampler: str | None = None,
    seed: int = 42,
) -> DataLoader:
    """Build a TensorDataset DataLoader (optionally with balanced sampling)."""
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float())
    if sampler == "balanced" and shuffle:
        labels = y.astype(int)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        per_class = {0: 1.0 / max(1, n_neg), 1: 1.0 / max(1, n_pos)}
        weights = np.array([per_class[int(li)] for li in labels], dtype=np.float64)
        gen = torch.Generator().manual_seed(seed)
        wsampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(labels),
            replacement=True,
            generator=gen,
        )
        return DataLoader(ds, batch_size=batch_size, sampler=wsampler, num_workers=0, drop_last=False)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False,
    )


def train_head(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    cfg: TrainHeadConfig,
    ckpt_dir: str | Path,
    *,
    resume: str | None = "auto",
) -> dict[str, object]:
    """Train the ESM-2 MLP head with resumable checkpoints.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : numpy.ndarray
        Train / val embeddings and binary labels.
    cfg : TrainHeadConfig
        Training configuration.
    ckpt_dir : str or Path
        Directory for ``last.pt`` and ``best.pt``.
    resume : str or None
        ``"auto"`` to load ``last.pt`` if present (default); ``None`` or
        ``""`` to start fresh; otherwise an explicit checkpoint path.

    Returns
    -------
    dict
        Best validation metrics, training time, and best-checkpoint path.
    """
    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")
    device = accelerator.device
    is_main = accelerator.is_main_process

    ckpt_dir = Path(ckpt_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if is_main:
        LOG.info(
            "Training head: in_dim=%d hidden=%d epochs=%d batch=%d sampler=%s",
            cfg.in_dim, cfg.hidden, cfg.epochs, cfg.batch_size, cfg.sampler,
        )

    train_loader = _make_loader(
        X_train, y_train,
        batch_size=cfg.batch_size, shuffle=True,
        sampler=cfg.sampler if cfg.sampler == "balanced" else None,
        seed=cfg.seed,
    )
    val_loader = _make_loader(X_val, y_val, batch_size=cfg.batch_size, shuffle=False)

    model = ESM2HeadMLP(in_dim=cfg.in_dim, hidden=cfg.hidden, dropout=cfg.dropout, norm=cfg.norm)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32) if cfg.sampler == "class_weight" else torch.tensor([1.0])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    start_epoch = 0
    global_step = 0
    best_val_auprc = -1.0
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"

    if resume == "auto" and last_path.exists():
        if is_main:
            LOG.info("Resuming from %s", last_path)
        state = load_checkpoint(last_path)
        accelerator.unwrap_model(model).load_state_dict(state.model_state)
        optimizer.load_state_dict(state.optimizer_state)
        if state.scheduler_state is not None:
            scheduler.load_state_dict(state.scheduler_state)
        start_epoch = state.epoch
        global_step = state.global_step
        best_val_auprc = state.best_val_metric
        restore_rng_state(state.rng_python, state.rng_numpy, state.rng_torch, state.rng_torch_cuda)

    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += loss.item() * y.size(0)
            n_seen += y.size(0)

            if is_main and global_step % cfg.ckpt_steps == 0:
                py, npr, th, th_cuda = gather_rng_state()
                save_checkpoint(
                    last_path,
                    CheckpointState(
                        model_state=accelerator.unwrap_model(model).state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        scheduler_state=scheduler.state_dict(),
                        scaler_state=None,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_metric=best_val_auprc,
                        rng_python=py, rng_numpy=npr, rng_torch=th, rng_torch_cuda=th_cuda,
                    ),
                )

        avg_loss = running_loss / max(1, n_seen)
        if is_main:
            inner = accelerator.unwrap_model(model)
            val_auprc, val_auroc = _evaluate(inner, val_loader, device=str(device))
            LOG.info(
                "epoch=%d step=%d loss=%.4f val_AUPRC=%.4f val_AUROC=%.4f",
                epoch, global_step, avg_loss, val_auprc, val_auroc,
            )
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                epochs_no_improve = 0
                py, npr, th, th_cuda = gather_rng_state()
                save_checkpoint(
                    best_path,
                    CheckpointState(
                        model_state=accelerator.unwrap_model(model).state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        scheduler_state=scheduler.state_dict(),
                        scaler_state=None,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_metric=best_val_auprc,
                        rng_python=py, rng_numpy=npr, rng_torch=th, rng_torch_cuda=th_cuda,
                    ),
                )
                LOG.info("New best val AUPRC=%.4f -> %s", best_val_auprc, best_path)
            else:
                epochs_no_improve += 1

            py, npr, th, th_cuda = gather_rng_state()
            save_checkpoint(
                last_path,
                CheckpointState(
                    model_state=accelerator.unwrap_model(model).state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    scaler_state=None,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_val_metric=best_val_auprc,
                    rng_python=py, rng_numpy=npr, rng_torch=th, rng_torch_cuda=th_cuda,
                ),
            )
            if epochs_no_improve >= cfg.early_stop_patience:
                LOG.info("Early stop at epoch=%d (no AUPRC improvement for %d epochs)", epoch, epochs_no_improve)
                break

    elapsed = time.time() - t_start
    return {
        "best_val_auprc": best_val_auprc,
        "epochs_run": epoch + 1 - start_epoch,
        "global_steps": global_step,
        "train_seconds": elapsed,
        "best_path": str(best_path),
        "last_path": str(last_path),
    }
