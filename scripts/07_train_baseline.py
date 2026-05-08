#!/usr/bin/env python3
"""Phase 7: train logistic-regression and random-forest baselines on the proposal-listed handcrafted features.

Inputs
------
- ``data/processed/features_handcrafted.npz`` (Phase 5 output)

Outputs
-------
- ``checkpoints/baseline/lr.joblib`` — best logistic-regression model.
- ``checkpoints/baseline/rf.joblib`` — best random-forest model.
- ``checkpoints/baseline/baseline_meta.json`` — best params + val metrics.
- ``reports/tables/baseline_val_metrics.csv`` — val AUPRC/AUROC per model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.models.baseline import train_logistic_regression, train_random_forest  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402
from imc.utils.seed import set_seed  # noqa: E402

LOG = get_logger("imc.scripts.train_baseline", log_file=ROOT / "logs" / "07_train_baseline.log")


def main() -> None:
    """Entry point: load features, train LR and RF, persist models + metrics."""
    cfg_data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    cfg_base = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text())
    set_seed(int(cfg_base["seed"]))

    processed_dir = ROOT / cfg_data["paths"]["processed_dir"]
    models_dir = ensure_dir(ROOT / cfg_base["paths"]["models_dir"])
    tables_dir = ensure_dir(ROOT / "reports" / "tables")

    npz = np.load(processed_dir / "features_handcrafted.npz", allow_pickle=True)
    X = npz["X"]
    y = npz["y"]
    splits = npz["split"].astype(str)
    LOG.info("Loaded features X=%s y=%s, splits=%s", X.shape, y.shape, np.unique(splits, return_counts=True))

    train_mask = splits == "train"
    val_mask = splits == "val"
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    LOG.info("Train: X=%s y=%s | Val: X=%s y=%s", X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    LOG.info("Training logistic regression with C grid %s ...", cfg_base["logistic_regression"]["C_grid"])
    lr = train_logistic_regression(
        X_train, y_train, X_val, y_val,
        C_grid=cfg_base["logistic_regression"]["C_grid"],
        max_iter=int(cfg_base["logistic_regression"]["max_iter"]),
        seed=int(cfg_base["seed"]),
    )
    joblib.dump(lr.model, models_dir / "lr.joblib")
    LOG.info("Saved best LR (C=%s) val AUPRC=%.4f AUROC=%.4f", lr.best_params, lr.val_auprc, lr.val_auroc)

    LOG.info("Training random forest with grid max_depth=%s min_samples_leaf=%s ...",
             cfg_base["random_forest"]["max_depth_grid"], cfg_base["random_forest"]["min_samples_leaf_grid"])
    rf = train_random_forest(
        X_train, y_train, X_val, y_val,
        n_estimators=int(cfg_base["random_forest"]["n_estimators"]),
        max_depth_grid=cfg_base["random_forest"]["max_depth_grid"],
        min_samples_leaf_grid=cfg_base["random_forest"]["min_samples_leaf_grid"],
        n_jobs=int(cfg_base["random_forest"]["n_jobs"]),
        seed=int(cfg_base["seed"]),
    )
    joblib.dump(rf.model, models_dir / "rf.joblib")
    LOG.info("Saved best RF (%s) val AUPRC=%.4f AUROC=%.4f", rf.best_params, rf.val_auprc, rf.val_auroc)

    meta = {
        "lr": {
            "best_params": lr.best_params,
            "val_auprc": lr.val_auprc,
            "val_auroc": lr.val_auroc,
            "train_seconds": lr.train_seconds,
            "n_train": lr.n_train,
        },
        "rf": {
            "best_params": rf.best_params,
            "val_auprc": rf.val_auprc,
            "val_auroc": rf.val_auroc,
            "train_seconds": rf.train_seconds,
            "n_train": rf.n_train,
        },
    }
    (models_dir / "baseline_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    metrics_df = pd.DataFrame(
        [
            {"model": "logistic_regression", "val_auprc": lr.val_auprc, "val_auroc": lr.val_auroc,
             "train_seconds": lr.train_seconds, "best_params": json.dumps(lr.best_params, default=str)},
            {"model": "random_forest", "val_auprc": rf.val_auprc, "val_auroc": rf.val_auroc,
             "train_seconds": rf.train_seconds, "best_params": json.dumps(rf.best_params, default=str)},
        ]
    )
    metrics_df.to_csv(tables_dir / "baseline_val_metrics.csv", index=False)
    LOG.info("Wrote %s", tables_dir / "baseline_val_metrics.csv")


if __name__ == "__main__":
    main()
