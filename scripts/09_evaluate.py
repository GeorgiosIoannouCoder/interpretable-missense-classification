#!/usr/bin/env python3
"""Phase 9: evaluate every model on val and test (with bootstrap 95% CIs).

For each model produces:

- Val and test AUROC, AUPRC, F1 (with 95% CIs), accuracy, precision,
  recall, specificity, confusion matrix.
- Per-gene AUROC distribution on the test split (Fig 6 source).
- Operating point chosen on val and applied to test.
- Computational-simplicity columns (training time, inference latency,
  parameter count, model-on-disk size) — populates the proposal's fourth
  tradeoff axis.

Outputs
-------
- ``reports/tables/results.csv`` — wide per-model metrics with CIs.
- ``reports/tables/per_gene_auroc.csv`` — long table of (model, gene, auroc).
- ``reports/tables/efficiency.csv`` — Table A1 (computational simplicity).
- ``reports/tables/test_predictions.parquet`` — variation_id + per-model
  scores (used by Phase 10 + Fig 5 UMAP).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.eval.metrics import best_threshold_by_f1, evaluate_scores, per_gene_auroc  # noqa: E402
from imc.models.head import ESM2HeadMLP  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402

LOG = get_logger("imc.scripts.evaluate", log_file=ROOT / "logs" / "09_evaluate.log")


def _file_size_mb(p: Path) -> float:
    """Return file size in MiB if the file exists, else 0.0."""
    return float(p.stat().st_size / (1024 * 1024)) if p.exists() else 0.0


def _measure_inference_latency_sklearn(model, X_sample: np.ndarray) -> float:
    """Measure single-row inference latency in milliseconds for an sklearn model."""
    model.predict_proba(X_sample[:1])
    n = min(64, X_sample.shape[0])
    t0 = time.time()
    for i in range(n):
        model.predict_proba(X_sample[i : i + 1])
    return float(((time.time() - t0) / n) * 1000.0)


def _measure_inference_latency_torch(model, X_sample: np.ndarray, device: str) -> float:
    """Measure single-row inference latency in milliseconds for a torch model."""
    with torch.inference_mode():
        x = torch.from_numpy(X_sample[:1]).to(device)
        _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        n = min(64, X_sample.shape[0])
        t0 = time.time()
        for i in range(n):
            xi = torch.from_numpy(X_sample[i : i + 1]).to(device)
            _ = model(xi)
        if device == "cuda":
            torch.cuda.synchronize()
    return float(((time.time() - t0) / n) * 1000.0)


def main() -> None:
    """Entry point: score every model on val/test, write all evaluation artifacts."""
    cfg_data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    cfg_base = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text())
    cfg_esm = yaml.safe_load((ROOT / "configs" / "esm2.yaml").read_text())

    processed_dir = ROOT / cfg_data["paths"]["processed_dir"]
    emb_dir = ROOT / cfg_esm["extraction"]["out_dir"]
    tables_dir = ensure_dir(ROOT / "reports" / "tables")

    splits = pd.read_parquet(processed_dir / "clinvar_split.parquet")[
        ["variation_id", "gene", "label", "split", "review_stars",
         "chrom", "pos", "ref_nt", "alt_nt", "uniprot_acc", "position_aa", "ref_aa", "alt_aa"]
    ]
    splits["variation_id"] = splits["variation_id"].astype(str)
    LOG.info("Loaded %d split rows", len(splits))

    feats = np.load(processed_dir / "features_handcrafted.npz", allow_pickle=True)
    X_all = feats["X"]
    var_id_feats = feats["variation_id"].astype(str)
    feat_split = feats["split"].astype(str)

    feats_idx = pd.DataFrame({"variation_id": var_id_feats}).reset_index().rename(columns={"index": "row"})
    splits_with_row = splits.merge(feats_idx, on="variation_id", how="inner")
    LOG.info("Joined splits + features: %d rows", len(splits_with_row))

    val_mask_arr = (feat_split == "val")
    test_mask_arr = (feat_split == "test")
    X_val = X_all[val_mask_arr]
    X_test = X_all[test_mask_arr]
    var_id_val = var_id_feats[val_mask_arr]
    var_id_test = var_id_feats[test_mask_arr]

    val_df = splits_with_row[splits_with_row["split"] == "val"].copy()
    test_df = splits_with_row[splits_with_row["split"] == "test"].copy()

    LR_PATH = ROOT / cfg_base["paths"]["models_dir"] / "lr.joblib"
    RF_PATH = ROOT / cfg_base["paths"]["models_dir"] / "rf.joblib"
    HEAD_BEST = ROOT / cfg_esm["training"]["ckpt_dir"] / "best.pt"

    LOG.info("Loading sklearn baselines ...")
    lr = joblib.load(LR_PATH)
    rf = joblib.load(RF_PATH)

    LOG.info("Scoring LR/RF on val and test ...")
    s_val_lr = lr.predict_proba(X_val)[:, 1]
    s_val_rf = rf.predict_proba(X_val)[:, 1]
    s_test_lr = lr.predict_proba(X_test)[:, 1]
    s_test_rf = rf.predict_proba(X_test)[:, 1]
    val_lr_lat = _measure_inference_latency_sklearn(lr, X_val)
    val_rf_lat = _measure_inference_latency_sklearn(rf, X_val)
    LOG.info("LR latency=%.3f ms, RF latency=%.3f ms", val_lr_lat, val_rf_lat)

    LOG.info("Loading ESM-2 head best.pt and scoring ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = ESM2HeadMLP(
        in_dim=int(cfg_esm["esm2"]["hidden_dim"]),
        hidden=int(cfg_esm["head"]["hidden_dim"]),
        dropout=float(cfg_esm["head"]["dropout"]),
        norm=str(cfg_esm["head"]["norm"]),
    ).to(device)
    payload = torch.load(HEAD_BEST, map_location=device, weights_only=False)
    head.load_state_dict(payload["model"])
    head.eval()

    emb = pd.read_parquet(emb_dir / "embeddings.parquet")
    emb["variation_id"] = emb["variation_id"].astype(str)
    feat_cols = [c for c in emb.columns if c.startswith("e")]

    def _score_emb(varids: np.ndarray) -> np.ndarray:
        sub = emb.set_index("variation_id").loc[varids][feat_cols].to_numpy(dtype=np.float32)
        with torch.inference_mode():
            x = torch.from_numpy(sub).to(device)
            scores = torch.sigmoid(head(x)).detach().cpu().numpy()
        return scores

    s_val_head = _score_emb(var_id_val)
    s_test_head = _score_emb(var_id_test)
    head_lat_X = emb.set_index("variation_id").loc[var_id_test[:64]][feat_cols].to_numpy(dtype=np.float32)
    head_lat = _measure_inference_latency_torch(head, head_lat_X, device=device)
    LOG.info("ESM-2 head latency=%.3f ms", head_lat)

    y_val_arr = val_df["label"].to_numpy(dtype=int)
    y_test_arr = test_df["label"].to_numpy(dtype=int)

    test_df = test_df.assign(
        score_lr=s_test_lr,
        score_rf=s_test_rf,
        score_esm2_head=s_test_head,
    )

    def _eval(name: str, sv: np.ndarray, st: np.ndarray, train_seconds: float, model_size_mb: float, params: int, latency_ms: float) -> dict:
        threshold = best_threshold_by_f1(y_val_arr, sv)
        m_test = evaluate_scores(y_test_arr, st, threshold=threshold, n_boot=1000, seed=42)
        m_val = evaluate_scores(y_val_arr, sv, threshold=threshold, n_boot=1000, seed=42)
        return {
            "model": name,
            "operating_threshold": threshold,
            "val_n": m_val.n, "val_n_pos": m_val.n_pos, "val_n_neg": m_val.n_neg,
            "val_auroc": m_val.auroc, "val_auprc": m_val.auprc, "val_f1": m_val.f1,
            "val_accuracy": m_val.accuracy,
            "test_n": m_test.n, "test_n_pos": m_test.n_pos, "test_n_neg": m_test.n_neg,
            "test_auroc": m_test.auroc, "test_auroc_lo": m_test.auroc_ci[0], "test_auroc_hi": m_test.auroc_ci[1],
            "test_auprc": m_test.auprc, "test_auprc_lo": m_test.auprc_ci[0], "test_auprc_hi": m_test.auprc_ci[1],
            "test_f1": m_test.f1, "test_f1_lo": m_test.f1_ci[0], "test_f1_hi": m_test.f1_ci[1],
            "test_accuracy": m_test.accuracy,
            "test_precision": m_test.precision, "test_recall": m_test.recall, "test_specificity": m_test.specificity,
            "test_tn": m_test.confusion[0], "test_fp": m_test.confusion[1],
            "test_fn": m_test.confusion[2], "test_tp": m_test.confusion[3],
            "train_seconds": train_seconds,
            "model_size_mb": model_size_mb,
            "params": params,
            "single_inference_ms": latency_ms,
        }

    base_meta = json.loads((ROOT / cfg_base["paths"]["models_dir"] / "baseline_meta.json").read_text())
    head_meta = json.loads((ROOT / cfg_esm["training"]["ckpt_dir"] / "training_meta.json").read_text())
    lr_size = _file_size_mb(LR_PATH)
    rf_size = _file_size_mb(RF_PATH)
    head_size = _file_size_mb(HEAD_BEST)
    n_params_head = sum(p.numel() for p in head.parameters())
    n_params_lr = int(getattr(lr, "coef_", np.zeros((1, X_val.shape[1]))).size + getattr(lr, "intercept_", np.zeros(1)).size)
    n_params_rf = int(rf.n_estimators) * 1

    rows = []
    rows.append(_eval("logistic_regression", s_val_lr, s_test_lr, base_meta["lr"]["train_seconds"], lr_size, n_params_lr, val_lr_lat))
    rows.append(_eval("random_forest", s_val_rf, s_test_rf, base_meta["rf"]["train_seconds"], rf_size, n_params_rf, val_rf_lat))
    rows.append(_eval("esm2_head", s_val_head, s_test_head, head_meta["train_seconds"], head_size, n_params_head, head_lat))

    df = pd.DataFrame(rows)
    out_csv = tables_dir / "results.csv"
    df.to_csv(out_csv, index=False)
    LOG.info("Wrote %s", out_csv)

    eff_cols = ["model", "params", "model_size_mb", "train_seconds", "single_inference_ms"]
    df[eff_cols].to_csv(tables_dir / "efficiency.csv", index=False)
    LOG.info("Wrote %s", tables_dir / "efficiency.csv")

    LOG.info("Computing per-gene AUROC on test ...")
    test_genes = test_df["gene"].to_numpy()
    pg_rows = []
    for model_name, scores in [
        ("logistic_regression", s_test_lr),
        ("random_forest", s_test_rf),
        ("esm2_head", s_test_head),
    ]:
        per_g = per_gene_auroc(y_test_arr, scores, test_genes, min_per_class=5)
        for g, a in per_g.items():
            pg_rows.append({"model": model_name, "gene": g, "auroc": a})
    pg_df = pd.DataFrame(pg_rows)
    pg_df.to_csv(tables_dir / "per_gene_auroc.csv", index=False)
    LOG.info("Wrote %s (%d rows)", tables_dir / "per_gene_auroc.csv", len(pg_df))

    test_df.to_parquet(tables_dir / "test_predictions.parquet", index=False)
    LOG.info("Wrote %s", tables_dir / "test_predictions.parquet")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
