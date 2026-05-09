#!/usr/bin/env python3
"""Phase 16: random-forest ablation over the five proposal feature families.

Trains a ``RandomForestClassifier`` using the Phase-7 best hyperparameters on
each cumulative feature subset and each leave-one-feature-family-out subset,
then reports test-set AUROC / AUPRC and writes Appendix Figure A3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402
from imc.utils.seed import set_seed  # noqa: E402

LOG = get_logger("imc.scripts.ablation_features", log_file=ROOT / "logs" / "13_ablation_features.log")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_FAMILY_SLICES: dict[str, slice] = {
    "ref_aa": slice(0, 20),
    "alt_aa": slice(20, 40),
    "blosum62": slice(40, 41),
    "window": slice(41, 61),
    "norm_position": slice(61, 62),
}
_FAMILY_ORDER: tuple[str, ...] = ("ref_aa", "alt_aa", "blosum62", "window", "norm_position")


def _family_column_indices(families: list[str]) -> np.ndarray:
    """Return sorted unique column indices for a list of feature-family keys."""
    cols: list[int] = []
    for fam in families:
        sl = _FAMILY_SLICES[fam]
        cols.extend(range(sl.start, sl.stop))
    return np.unique(np.array(cols, dtype=np.int64))


def _build_rf(best_params: dict[str, object], n_estimators: int, n_jobs: int, seed: int) -> RandomForestClassifier:
    """Instantiate the tuned random forest from Phase 7 metadata."""
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=best_params["max_depth"],
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        class_weight="balanced",
        n_jobs=int(n_jobs),
        random_state=int(seed),
    )


def main_table() -> pd.DataFrame:
    """Fit ablation models and return a summary table."""
    cfg_data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    cfg_base = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text())
    set_seed(int(cfg_base["seed"]))

    processed_dir = ROOT / cfg_data["paths"]["processed_dir"]
    models_dir = ROOT / cfg_base["paths"]["models_dir"]
    meta = json.loads((models_dir / "baseline_meta.json").read_text())
    rf_best = meta["rf"]["best_params"]

    npz = np.load(processed_dir / "features_handcrafted.npz", allow_pickle=True)
    X = np.asarray(npz["X"], dtype=np.float32)
    y = np.asarray(npz["y"], dtype=np.int32)
    splits = np.asarray(npz["split"]).astype(str)
    train_m = splits == "train"
    test_m = splits == "test"
    X_tr, y_train = X[train_m], y[train_m]
    X_te, y_test = X[test_m], y[test_m]

    rows: list[dict[str, object]] = []
    specs: list[tuple[str, list[str]]] = []

    for k in range(1, len(_FAMILY_ORDER) + 1):
        fams = list(_FAMILY_ORDER[:k])
        name = " + ".join(fams)
        specs.append((name, fams))

    full_fams = list(_FAMILY_ORDER)
    for leave_out in _FAMILY_ORDER:
        fams = [f for f in full_fams if f != leave_out]
        specs.append((f"full_minus_{leave_out}", fams))

    for subset_name, fams in specs:
        cols = _family_column_indices(fams)
        rf = _build_rf(
            rf_best,
            n_estimators=int(cfg_base["random_forest"]["n_estimators"]),
            n_jobs=int(cfg_base["random_forest"]["n_jobs"]),
            seed=int(cfg_base["seed"]),
        )
        rf.fit(X_tr[:, cols], y_train)
        s_te = rf.predict_proba(X_te[:, cols])[:, 1]
        rows.append({
            "subset": subset_name,
            "n_features": int(cols.shape[0]),
            "families": ",".join(fams),
            "test_auroc": float(roc_auc_score(y_test, s_te)),
            "test_auprc": float(average_precision_score(y_test, s_te)),
        })
        LOG.info("%s | cols=%d test AUPRC=%.4f", subset_name, cols.shape[0], rows[-1]["test_auprc"])

    return pd.DataFrame(rows)


def fig_ablation(csv_path: Path, out_path: Path) -> None:
    """Horizontal bar chart of test AUPRC per subset (Figure A3)."""
    tbl = pd.read_csv(csv_path)
    order = tbl["test_auprc"].argsort()
    tbl = tbl.iloc[order]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(tbl))))
    y_pos = np.arange(len(tbl))
    ax.barh(y_pos, tbl["test_auprc"].to_numpy(), color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tbl["subset"].to_list(), fontsize=9)
    ax.set_xlabel("Test AUPRC")
    ax.set_title("Random forest feature-family ablation")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def main() -> None:
    """Entry point: run ablations, write CSV + appendix figure."""
    tables_dir = ensure_dir(ROOT / "reports" / "tables")
    fig_dir = ensure_dir(ROOT / "reports" / "figures")
    df = main_table()
    out_csv = tables_dir / "ablation_features.csv"
    df.to_csv(out_csv, index=False)
    LOG.info("Wrote %s", out_csv)
    fig_ablation(out_csv, fig_dir / "figA3_feature_ablation.pdf")


if __name__ == "__main__":
    main()
