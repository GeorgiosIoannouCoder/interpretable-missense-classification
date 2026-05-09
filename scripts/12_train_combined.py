#!/usr/bin/env python3
"""Phase 13: train combined handcrafted + ESM-2 MLP head with resumable checkpoints.

Concatenates the 62-d proposal handcrafted vector with the 1280-d ESM-2 residue
embedding (1342-d input) and optimizes the same two-layer MLP head architecture
as Phase 8.

Single GPU::

    python3 scripts/12_train_combined.py

Multi-GPU::

    accelerate launch --num_processes=N scripts/12_train_combined.py
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

from imc.features.handcrafted import feature_dim  # noqa: E402
from imc.models.combined import CombinedHead  # noqa: E402
from imc.training.train_head import TrainHeadConfig, train_head  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402
from imc.utils.seed import set_seed  # noqa: E402

LOG = get_logger("imc.scripts.train_combined", log_file=ROOT / "logs" / "12_train_combined.log")


def _load_combined_split_arrays(
    embeddings_parquet: Path,
    splits_parquet: Path,
    handcrafted_npz: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Join handcrafted features, embeddings, and splits; return per-split data.

    Parameters
    ----------
    embeddings_parquet : Path
        Phase 6 ``embeddings.parquet``.
    splits_parquet : Path
        Phase 4 ``clinvar_split.parquet``.
    handcrafted_npz : Path
        Phase 5 ``features_handcrafted.npz``.

    Returns
    -------
    dict[str, tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]
        Mapping ``split -> (variation_id, X_combined, y)``.
    """
    LOG.info("Loading embeddings from %s", embeddings_parquet)
    emb = pd.read_parquet(embeddings_parquet)
    emb["variation_id"] = emb["variation_id"].astype(str)
    feat_cols = [c for c in emb.columns if c.startswith("e")]

    LOG.info("Loading splits from %s", splits_parquet)
    sdf = pd.read_parquet(splits_parquet)[["variation_id", "label", "split"]]
    sdf["variation_id"] = sdf["variation_id"].astype(str)

    LOG.info("Loading handcrafted features from %s", handcrafted_npz)
    npz = np.load(handcrafted_npz, allow_pickle=True)
    h_var = npz["variation_id"].astype(str)
    X_h = np.asarray(npz["X"], dtype=np.float32)
    hand = pd.DataFrame({"variation_id": h_var, "h_row": np.arange(len(h_var), dtype=np.int64)})

    merged = emb.merge(sdf, on="variation_id", how="inner").merge(hand, on="variation_id", how="inner")
    LOG.info(
        "Joined handcrafted+embeddings+splits: %d rows (emb %d, splits %d)",
        len(merged),
        len(emb),
        len(sdf),
    )

    X_emb = merged[feat_cols].to_numpy(dtype=np.float32)
    X_hand = X_h[merged["h_row"].to_numpy()]
    X_cat = np.concatenate([X_hand, X_emb], axis=1).astype(np.float32, copy=False)
    y = merged["label"].to_numpy(dtype=np.float32)
    var_ids = merged["variation_id"].to_numpy()
    splits = merged["split"].to_numpy()

    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        m = splits == split
        out[split] = (var_ids[m], X_cat[m], y[m])
        _, Xs, ys = out[split]
        LOG.info("split=%s: X=%s y=%s pos=%d", split, Xs.shape, ys.shape, int((ys == 1).sum()))
    return out


def main() -> None:
    """Entry point: build concatenated features, train combined head, persist meta."""
    cfg_data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    cfg_esm = yaml.safe_load((ROOT / "configs" / "esm2.yaml").read_text())
    set_seed(int(cfg_esm["seed"]))

    processed_dir = ROOT / cfg_data["paths"]["processed_dir"]
    emb_dir = ROOT / cfg_esm["extraction"]["out_dir"]
    ckpt_dir = ensure_dir(ROOT / cfg_esm["combined_training"]["ckpt_dir"])
    tables_dir = ensure_dir(ROOT / "reports" / "tables")

    h_dim = feature_dim()
    e_dim = int(cfg_esm["esm2"]["hidden_dim"])
    in_dim = h_dim + e_dim

    arrays = _load_combined_split_arrays(
        embeddings_parquet=emb_dir / "embeddings.parquet",
        splits_parquet=processed_dir / "clinvar_split.parquet",
        handcrafted_npz=processed_dir / "features_handcrafted.npz",
    )
    _, X_train, y_train = arrays["train"]
    _, X_val, y_val = arrays["val"]

    cfg = TrainHeadConfig(
        in_dim=in_dim,
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
    result = train_head(
        X_train, y_train, X_val, y_val, cfg, ckpt_dir=ckpt_dir, resume="auto", model_cls=CombinedHead,
    )
    LOG.info("Done: %s", result)
    (ckpt_dir / "training_meta.json").write_text(json.dumps(result, indent=2, default=str))

    pd.DataFrame([
        {
            "model": "combined_head",
            "val_auprc": result["best_val_auprc"],
            "train_seconds": result["train_seconds"],
        },
    ]).to_csv(tables_dir / "combined_head_val_metrics.csv", index=False)


if __name__ == "__main__":
    main()
