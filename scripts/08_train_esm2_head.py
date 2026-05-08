#!/usr/bin/env python3
"""Phase 8: train the ESM-2 MLP classifier head with resumable checkpoints.

Single GPU::

    python3 scripts/08_train_esm2_head.py

Multi-GPU (no code changes)::

    accelerate launch --num_processes=N scripts/08_train_esm2_head.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.training.train_head import TrainHeadConfig, train_head  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402
from imc.utils.seed import set_seed  # noqa: E402

LOG = get_logger("imc.scripts.train_esm2_head", log_file=ROOT / "logs" / "08_train_esm2_head.log")


def _load_split_arrays(
    embeddings_parquet: Path,
    splits_parquet: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Join embeddings with split assignments and return per-split arrays.

    Parameters
    ----------
    embeddings_parquet : Path
        Path to ``embeddings.parquet`` (Phase 6 output).
    splits_parquet : Path
        Path to ``clinvar_split.parquet`` (Phase 4 output).

    Returns
    -------
    dict[str, tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]
        Mapping ``split -> (variation_id, X, y)``.
    """
    LOG.info("Loading embeddings from %s", embeddings_parquet)
    emb = pd.read_parquet(embeddings_parquet)
    LOG.info("Loading splits from %s", splits_parquet)
    sdf = pd.read_parquet(splits_parquet)[["variation_id", "label", "split"]]
    sdf["variation_id"] = sdf["variation_id"].astype(str)
    emb["variation_id"] = emb["variation_id"].astype(str)
    merged = emb.merge(sdf, on="variation_id", how="inner")
    LOG.info("Joined: %d rows (embeddings %d, splits %d)", len(merged), len(emb), len(sdf))

    feat_cols = [c for c in merged.columns if c.startswith("e")]
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        sub = merged[merged["split"] == split].reset_index(drop=True)
        var_id = sub["variation_id"].to_numpy()
        X = sub[feat_cols].to_numpy(dtype=np.float32)
        y = sub["label"].to_numpy(dtype=np.float32)
        out[split] = (var_id, X, y)
        LOG.info("split=%s: X=%s y=%s pos=%d", split, X.shape, y.shape, int((y == 1).sum()))
    return out


def main() -> None:
    """Entry point: load embeddings + splits, train head, persist meta."""
    cfg_data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    cfg_esm = yaml.safe_load((ROOT / "configs" / "esm2.yaml").read_text())
    set_seed(int(cfg_esm["seed"]))

    processed_dir = ROOT / cfg_data["paths"]["processed_dir"]
    emb_dir = ROOT / cfg_esm["extraction"]["out_dir"]
    ckpt_dir = ensure_dir(ROOT / cfg_esm["training"]["ckpt_dir"])
    tables_dir = ensure_dir(ROOT / "reports" / "tables")

    arrays = _load_split_arrays(
        embeddings_parquet=emb_dir / "embeddings.parquet",
        splits_parquet=processed_dir / "clinvar_split.parquet",
    )
    _, X_train, y_train = arrays["train"]
    _, X_val, y_val = arrays["val"]

    cfg = TrainHeadConfig(
        in_dim=int(cfg_esm["esm2"]["hidden_dim"]),
        hidden=int(cfg_esm["head"]["hidden_dim"]),
        dropout=float(cfg_esm["head"]["dropout"]),
        norm=str(cfg_esm["head"]["norm"]),
        lr=float(cfg_esm["training"]["lr"]),
        weight_decay=float(cfg_esm["training"]["weight_decay"]),
        epochs=int(cfg_esm["training"]["epochs"]),
        batch_size=int(cfg_esm["training"]["batch_size"]),
        sampler=str(cfg_esm["training"]["sampler"]),
        early_stop_patience=int(cfg_esm["training"]["early_stop_patience"]),
        ckpt_steps=int(cfg_esm["training"]["ckpt_steps"]),
        seed=int(cfg_esm["seed"]),
    )
    result = train_head(X_train, y_train, X_val, y_val, cfg, ckpt_dir=ckpt_dir, resume="auto")
    LOG.info("Done: %s", result)
    (ckpt_dir / "training_meta.json").write_text(json.dumps(result, indent=2, default=str))

    pd.DataFrame([
        {"model": "esm2_head", "val_auprc": result["best_val_auprc"], "train_seconds": result["train_seconds"]},
    ]).to_csv(tables_dir / "esm2_head_val_metrics.csv", index=False)


if __name__ == "__main__":
    main()
