#!/usr/bin/env python3
"""Phase 11: generate every figure and table needed for the final LaTeX report.

Outputs land in ``reports/figures/`` and ``reports/tables/``. Per the
instructions' "2-4 graphical items in Results" cap, the main-text figures
are 2/3/6 (ROC+PR, confusion matrices, per-gene AUROC) plus Table 2
(test-set metrics). Table 1 (dataset funnel) lives in Methods. Figures
4/5/7/8 plus Table A1 are appendix material.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.features.handcrafted import HandcraftedConfig, feature_names  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402

LOG = get_logger("imc.scripts.make_figures", log_file=ROOT / "logs" / "11_make_figures.log")

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

MODEL_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "esm2_head": "ESM-2 head",
    "alphamissense": "AlphaMissense",
    "cadd_phred": "CADD",
}
MODEL_ORDER: list[str] = list(MODEL_LABELS.keys())
PALETTE = sns.color_palette("colorblind", n_colors=len(MODEL_ORDER))


def _load_test_predictions(tables_dir: Path) -> pd.DataFrame:
    """Load Phase 9/10 enriched test predictions parquet."""
    df = pd.read_parquet(tables_dir / "test_predictions.parquet")
    return df


def _normalize_for_curves(df: pd.DataFrame, model_key: str, score_col: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (y, scores, label) for one model on the test set, dropping NaNs."""
    sub = df[["label", score_col]].dropna()
    return sub["label"].to_numpy(int), sub[score_col].to_numpy(float), MODEL_LABELS[model_key]


def fig_roc_pr(test_pred: pd.DataFrame, out_path: Path) -> None:
    """Figure 2: ROC and PR curves for all models on the test set."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    score_cols = {
        "logistic_regression": "score_lr",
        "random_forest": "score_rf",
        "esm2_head": "score_esm2_head",
        "alphamissense": "am_pathogenicity",
        "cadd_phred": "cadd_phred",
    }
    for color, key in zip(PALETTE, MODEL_ORDER):
        col = score_cols[key]
        if col not in test_pred.columns:
            continue
        y, s, label = _normalize_for_curves(test_pred, key, col)
        if len(y) == 0:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, label=f"{label} (n={len(y)})", color=color, linewidth=1.8)
        axes[1].plot(rec, prec, label=f"{label} (n={len(y)})", color=color, linewidth=1.8)
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves (test set)")
    axes[0].legend(loc="lower right")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curves (test set)")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def fig_confusion(test_pred: pd.DataFrame, results: pd.DataFrame, out_path: Path) -> None:
    """Figure 3: confusion-matrix grid for LR, RF, ESM-2 head."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4))
    pairs = [
        ("logistic_regression", "score_lr"),
        ("random_forest", "score_rf"),
        ("esm2_head", "score_esm2_head"),
    ]
    for ax, (key, col) in zip(axes, pairs):
        thr = float(results[results["model"] == key]["operating_threshold"].iloc[0])
        sub = test_pred[["label", col]].dropna()
        y = sub["label"].to_numpy(int)
        s = sub[col].to_numpy(float)
        preds = (s >= thr).astype(int)
        cm = confusion_matrix(y, preds, labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["pred neg", "pred pos"], yticklabels=["true neg", "true pos"],
            annot_kws={"size": 11},
        )
        ax.set_title(f"{MODEL_LABELS[key]}\nthreshold={thr:.3f}, n={len(y)}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def fig_per_gene_auroc(per_gene_csv: Path, out_path: Path) -> None:
    """Figure 6: per-gene test AUROC distribution as a boxplot."""
    df = pd.read_csv(per_gene_csv)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df = df[df["model_label"].notna()]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    order = [MODEL_LABELS[m] for m in ["logistic_regression", "random_forest", "esm2_head"] if m in df["model"].unique()]
    sns.boxplot(
        data=df, x="model_label", y="auroc",
        order=order, ax=ax,
        width=0.55, fliersize=2,
    )
    n_per = df.groupby("model_label").size()
    for i, lbl in enumerate(order):
        ax.text(i, 0.05, f"n={int(n_per.get(lbl, 0))} genes", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Per-gene AUROC (test)")
    ax.set_title("Cross-gene generalization: per-gene test-set AUROC distribution")
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def fig_pipeline_schematic(out_path: Path) -> None:
    """Figure 1: small pipeline schematic for the Methods inset."""
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    ax.axis("off")
    boxes = [
        ("ClinVar\nvariant_summary", 0.05),
        ("Filter germline\nmissense P/LP vs B/LB", 0.20),
        ("Map to UniProt\n(reviewed Swiss-Prot)", 0.38),
        ("Gene-disjoint\n70/10/20 split", 0.55),
        ("Handcrafted features\n(LR, RF)", 0.72),
        ("ESM-2 (650M)\nresidue embeddings\n+ MLP head", 0.88),
    ]
    for label, x in boxes:
        ax.add_patch(plt.Rectangle((x - 0.07, 0.30), 0.14, 0.55, facecolor="#E8F0FE", edgecolor="black"))
        ax.text(x, 0.575, label, ha="center", va="center", fontsize=9.5)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + 0.07
        x1 = boxes[i + 1][1] - 0.07
        ax.annotate(
            "", xy=(x1, 0.575), xytext=(x0, 0.575),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def fig_feature_importance(out_path: Path, lr_path: Path, rf_path: Path) -> None:
    """Figure 4 (Appendix): LR coefficients and RF Gini importances."""
    lr = joblib.load(lr_path)
    rf = joblib.load(rf_path)
    names = feature_names(HandcraftedConfig())
    lr_coef = np.asarray(lr.coef_).ravel()
    rf_imp = np.asarray(rf.feature_importances_)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    top_lr = np.argsort(np.abs(lr_coef))[::-1][:15]
    coef_top = lr_coef[top_lr]
    names_lr = [names[i] for i in top_lr]
    colors = ["#377eb8" if c >= 0 else "#e41a1c" for c in coef_top]
    axes[0].barh(range(len(names_lr))[::-1], coef_top, color=colors)
    axes[0].set_yticks(range(len(names_lr))[::-1])
    axes[0].set_yticklabels(names_lr, fontsize=9)
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("LR coefficient (positive = pathogenic)")
    axes[0].set_title("Top-15 LR coefficients by |magnitude|")

    top_rf = np.argsort(rf_imp)[::-1][:15]
    imp_top = rf_imp[top_rf]
    names_rf = [names[i] for i in top_rf]
    axes[1].barh(range(len(names_rf))[::-1], imp_top, color="#4daf4a")
    axes[1].set_yticks(range(len(names_rf))[::-1])
    axes[1].set_yticklabels(names_rf, fontsize=9)
    axes[1].set_xlabel("RF Gini importance")
    axes[1].set_title("Top-15 RF Gini-importance features")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def fig_umap(test_pred: pd.DataFrame, embeddings_parquet: Path, out_path: Path, sample: int = 5000) -> None:
    """Figure 5 (Appendix): UMAP of ESM-2 test embeddings colored by label."""
    import umap

    LOG.info("Loading embeddings for UMAP ...")
    emb = pd.read_parquet(embeddings_parquet)
    emb["variation_id"] = emb["variation_id"].astype(str)
    test_pred["variation_id"] = test_pred["variation_id"].astype(str)

    rng = np.random.default_rng(42)
    test_idx = test_pred.index.to_numpy()
    if len(test_idx) > sample:
        test_idx = rng.choice(test_idx, size=sample, replace=False)
    sub_pred = test_pred.loc[test_idx, ["variation_id", "label"]]
    sub_emb = emb.set_index("variation_id").loc[sub_pred["variation_id"].to_numpy()]
    feat_cols = [c for c in sub_emb.columns if c.startswith("e")]
    X = sub_emb[feat_cols].to_numpy(dtype=np.float32)
    LOG.info("UMAP-fitting %d points ...", X.shape[0])

    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, n_jobs=1)
    Z = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    labels = sub_pred["label"].to_numpy(int)
    for label_val, color, name in [(0, "#377eb8", "Benign / LB"), (1, "#e41a1c", "Pathogenic / LP")]:
        mask = labels == label_val
        ax.scatter(Z[mask, 0], Z[mask, 1], s=4, alpha=0.5, color=color, label=f"{name} (n={mask.sum()})")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("ESM-2 residue embeddings of test variants (UMAP, cosine)")
    ax.legend(markerscale=3, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def fig_studiedness(test_pred: pd.DataFrame, splits_parquet: Path, out_path: Path) -> None:
    """Figure 7 (Appendix): per-test-gene AUROC vs gene 'studiedness' proxy.

    We use total ClinVar variant count for the gene (training + test) as a
    studiedness proxy: well-studied genes have many ClinVar variants and are
    over-represented in ESM-2's pretraining corpus.
    """
    splits_df = pd.read_parquet(splits_parquet)
    studiedness = splits_df.groupby("gene").size().rename("variants_per_gene").reset_index()
    studiedness["log_n_variants"] = np.log1p(studiedness["variants_per_gene"])

    score_cols = {
        "logistic_regression": "score_lr",
        "random_forest": "score_rf",
        "esm2_head": "score_esm2_head",
    }

    rows = []
    test_local = test_pred.copy()
    test_local["gene"] = test_local["gene"].astype(str)
    for gene, sub in test_local.groupby("gene"):
        if int((sub["label"] == 1).sum()) < 5 or int((sub["label"] == 0).sum()) < 5:
            continue
        for key, col in score_cols.items():
            if col not in sub.columns:
                continue
            try:
                from sklearn.metrics import roc_auc_score

                rows.append({
                    "gene": gene, "model": key,
                    "auroc": float(roc_auc_score(sub["label"].astype(int), sub[col].astype(float))),
                })
            except (ValueError, KeyError):
                continue

    if not rows:
        LOG.warning("No genes with both classes >= 5 for studiedness plot; skipping.")
        return

    df = pd.DataFrame(rows).merge(studiedness, on="gene", how="left")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for color, key in zip(PALETTE[:3], ["logistic_regression", "random_forest", "esm2_head"]):
        sub = df[df["model"] == key].sort_values("log_n_variants")
        ax.scatter(sub["log_n_variants"], sub["auroc"], s=12, alpha=0.6, color=color, label=MODEL_LABELS[key])
        if len(sub) >= 10:
            x = sub["log_n_variants"].to_numpy()
            y = sub["auroc"].to_numpy()
            try:
                z = np.polyfit(x, y, 1)
                xx = np.linspace(x.min(), x.max(), 100)
                ax.plot(xx, np.poly1d(z)(xx), color=color, linewidth=1.5, linestyle="--")
            except np.linalg.LinAlgError:
                pass
    ax.set_xlabel("Gene 'studiedness' proxy: log(1 + ClinVar variants in gene)")
    ax.set_ylabel("Per-gene test AUROC")
    ax.set_title("Per-gene AUROC vs gene studiedness (n=%d genes)" % df["gene"].nunique())
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def fig_review_status_sensitivity(test_pred: pd.DataFrame, splits_parquet: Path, out_path: Path) -> None:
    """Figure 8 (Appendix): metrics on full vs >=2-star vs expert-reviewed (>=3-star) test subsets."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    test_pred = test_pred.copy()
    test_pred["variation_id"] = test_pred["variation_id"].astype(str)
    if "review_stars" not in test_pred.columns:
        splits_df = pd.read_parquet(splits_parquet)[["variation_id", "review_stars"]]
        splits_df["variation_id"] = splits_df["variation_id"].astype(str)
        df = test_pred.merge(splits_df, on="variation_id", how="left")
    else:
        df = test_pred

    score_cols = {
        "logistic_regression": "score_lr",
        "random_forest": "score_rf",
        "esm2_head": "score_esm2_head",
    }

    rows = []
    for stratum_name, mask in [
        ("full", pd.Series(True, index=df.index)),
        (">=2-star", df["review_stars"] >= 2),
        ("expert (>=3-star)", df["review_stars"] >= 3),
    ]:
        sub = df[mask]
        n_pos = int((sub["label"] == 1).sum()); n_neg = int((sub["label"] == 0).sum())
        if n_pos < 10 or n_neg < 10:
            continue
        for key, col in score_cols.items():
            ss = sub[["label", col]].dropna()
            if len(ss) == 0 or ss["label"].nunique() < 2:
                continue
            rows.append({
                "stratum": stratum_name, "n": int(len(ss)),
                "model": MODEL_LABELS[key],
                "AUROC": float(roc_auc_score(ss["label"], ss[col])),
                "AUPRC": float(average_precision_score(ss["label"], ss[col])),
            })

    df_rows = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(
        data=df_rows, x="stratum", y="AUROC", hue="model",
        order=["full", ">=2-star", "expert (>=3-star)"],
        palette=[PALETTE[0], PALETTE[1], PALETTE[2]],
        ax=axes[0],
    )
    axes[0].set_ylim(0.6, 1.0); axes[0].set_title("Test AUROC by review-status stratum")
    sns.barplot(
        data=df_rows, x="stratum", y="AUPRC", hue="model",
        order=["full", ">=2-star", "expert (>=3-star)"],
        palette=[PALETTE[0], PALETTE[1], PALETTE[2]],
        ax=axes[1],
    )
    axes[1].set_ylim(0.4, 1.0); axes[1].set_title("Test AUPRC by review-status stratum")
    n_by_stratum = df_rows.drop_duplicates("stratum").set_index("stratum")["n"]
    for ax in axes:
        ax.set_xlabel("Review-status stratum  " + " | ".join(f"{k}: n={v}" for k, v in n_by_stratum.items()))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def main() -> None:
    """Entry point: load all artifacts and write every figure / table."""
    cfg_data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    cfg_base = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text())
    cfg_esm = yaml.safe_load((ROOT / "configs" / "esm2.yaml").read_text())
    processed_dir = ROOT / cfg_data["paths"]["processed_dir"]
    tables_dir = ensure_dir(ROOT / "reports" / "tables")
    fig_dir = ensure_dir(ROOT / "reports" / "figures")

    test_pred = _load_test_predictions(tables_dir)
    results = pd.read_csv(tables_dir / "results.csv")

    fig_pipeline_schematic(fig_dir / "fig1_pipeline.pdf")
    fig_roc_pr(test_pred, fig_dir / "fig2_roc_pr.pdf")
    fig_confusion(test_pred, results, fig_dir / "fig3_confusion.pdf")
    fig_per_gene_auroc(tables_dir / "per_gene_auroc.csv", fig_dir / "fig6_per_gene_auroc.pdf")

    fig_feature_importance(
        fig_dir / "fig4_feature_importance.pdf",
        ROOT / cfg_base["paths"]["models_dir"] / "lr.joblib",
        ROOT / cfg_base["paths"]["models_dir"] / "rf.joblib",
    )
    try:
        fig_umap(
            test_pred=test_pred,
            embeddings_parquet=ROOT / cfg_esm["extraction"]["out_dir"] / "embeddings.parquet",
            out_path=fig_dir / "fig5_umap.pdf",
        )
    except Exception as exc:
        LOG.warning("UMAP figure failed: %s", exc)
    try:
        fig_studiedness(
            test_pred=test_pred,
            splits_parquet=processed_dir / "clinvar_split.parquet",
            out_path=fig_dir / "fig7_studiedness.pdf",
        )
    except Exception as exc:
        LOG.warning("Studiedness figure failed: %s", exc)
    try:
        fig_review_status_sensitivity(
            test_pred=test_pred,
            splits_parquet=processed_dir / "clinvar_split.parquet",
            out_path=fig_dir / "fig8_review_status.pdf",
        )
    except Exception as exc:
        LOG.warning("Review-status figure failed: %s", exc)

    LOG.info("All figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
