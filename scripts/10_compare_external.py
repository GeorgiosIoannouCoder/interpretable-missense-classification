#!/usr/bin/env python3
"""Phase 10: AlphaMissense + CADD comparison on test variants only.

Both tools are reference points (per the proposal), not targets. We restrict
the comparison to the **same** test variants used to evaluate our models so
all numbers are directly comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.eval.external import load_alphamissense_for_variants, load_cadd_for_variants  # noqa: E402
from imc.eval.metrics import best_threshold_by_f1, evaluate_scores, paired_bootstrap_difference  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402

LOG = get_logger("imc.scripts.compare_external", log_file=ROOT / "logs" / "10_compare_external.log")


def _merge_external_pairwise_rows(tables_dir: Path, enriched: pd.DataFrame) -> None:
    """Refresh paired-bootstrap rows for ESM-2 vs AlphaMissense / CADD on the test table."""
    specs: list[tuple[str, str, str, str]] = [
        ("esm2_head", "alphamissense", "score_esm2_head", "am_pathogenicity"),
        ("esm2_head", "cadd_phred", "score_esm2_head", "cadd_phred"),
    ]
    additions: list[dict[str, object]] = []
    for model_a, model_b, col_a, col_b in specs:
        if col_a not in enriched.columns or col_b not in enriched.columns:
            continue
        sub = enriched[["label", col_a, col_b]].dropna()
        yv = sub["label"].to_numpy(dtype=int)
        sa = sub[col_a].to_numpy(dtype=float)
        sb = sub[col_b].to_numpy(dtype=float)
        for metric_name in ("auroc", "auprc"):
            r = paired_bootstrap_difference(yv, sa, sb, metric=metric_name, n_boot=1000, seed=42)
            additions.append({
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric_name,
                "n": int(len(yv)),
                **r,
            })
    if not additions:
        return
    path = tables_dir / "pairwise_tests.csv"
    base = pd.read_csv(path) if path.exists() else pd.DataFrame()
    keys_drop = {(a, b) for a, b, _, _ in specs}
    if not base.empty:
        mask = ~base.apply(lambda row: (row["model_a"], row["model_b"]) in keys_drop, axis=1)
        base = base[mask]
    out = pd.concat([base, pd.DataFrame(additions)], ignore_index=True)
    out.to_csv(path, index=False)
    LOG.info("Updated %s with external pairwise rows (%d total)", path, len(out))


def main() -> None:
    """Entry point: join AlphaMissense + CADD with test set, append to results.csv."""
    cfg_data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    external_dir = ROOT / cfg_data["paths"]["external_dir"]
    tables_dir = ensure_dir(ROOT / "reports" / "tables")

    test_pred = pd.read_parquet(tables_dir / "test_predictions.parquet")
    nd = int(test_pred["variation_id"].duplicated().sum())
    if nd:
        LOG.warning("Input test_predictions has %d duplicate variation_id rows; keeping first", nd)
        test_pred = test_pred.drop_duplicates(subset=["variation_id"], keep="first").reset_index(drop=True)
    LOG.info("Loaded %d test variants", len(test_pred))

    base_keys = test_pred[["variation_id", "chrom", "pos", "ref_nt", "alt_nt", "label", "gene"]].copy()
    base_keys["pos"] = base_keys["pos"].astype(int)
    base_keys["chrom"] = base_keys["chrom"].astype(str)
    if int(base_keys["variation_id"].duplicated().sum()):
        n0 = len(base_keys)
        base_keys = base_keys.drop_duplicates(subset=["variation_id"], keep="first")
        LOG.warning("Deduplicated base_keys by variation_id: %d -> %d", n0, len(base_keys))
    am = load_alphamissense_for_variants(external_dir / "AlphaMissense_hg38.tsv.gz", base_keys)
    cadd = load_cadd_for_variants(external_dir / "whole_genome_SNVs.tsv.gz", base_keys)

    dup_am = int(am["variation_id"].duplicated().sum())
    dup_cd = int(cadd["variation_id"].duplicated().sum())
    if dup_am or dup_cd:
        LOG.warning("Dropping duplicate variation_id rows before merge (am=%d, cadd=%d)", dup_am, dup_cd)
    am = am.drop_duplicates(subset=["variation_id"], keep="first")
    cadd = cadd.drop_duplicates(subset=["variation_id"], keep="first")
    joined = base_keys.merge(
        am[["variation_id", "am_pathogenicity", "am_class"]],
        on="variation_id",
        how="left",
    ).merge(
        cadd[["variation_id", "cadd_phred"]],
        on="variation_id",
        how="left",
    )
    LOG.info(
        "Coverage on test set: AlphaMissense=%d/%d (%.1f%%), CADD=%d/%d (%.1f%%)",
        int(joined["am_pathogenicity"].notna().sum()), len(joined),
        100.0 * joined["am_pathogenicity"].notna().mean(),
        int(joined["cadd_phred"].notna().sum()), len(joined),
        100.0 * joined["cadd_phred"].notna().mean(),
    )

    enriched_path = tables_dir / "test_predictions.parquet"
    enriched = test_pred.merge(
        joined[["variation_id", "am_pathogenicity", "am_class", "cadd_phred"]],
        on="variation_id",
        how="left",
    )
    enriched.to_parquet(enriched_path, index=False)
    LOG.info("Wrote enriched %s", enriched_path)
    _merge_external_pairwise_rows(tables_dir, enriched)

    rows = []

    am_mask = joined["am_pathogenicity"].notna()
    if am_mask.any():
        y = joined.loc[am_mask, "label"].to_numpy(dtype=int)
        s = joined.loc[am_mask, "am_pathogenicity"].to_numpy(dtype=float)
        threshold = best_threshold_by_f1(y, s)
        m = evaluate_scores(y, s, threshold=threshold, n_boot=1000, seed=42)
        rows.append({
            "model": "alphamissense",
            "operating_threshold": threshold,
            "test_n": m.n, "test_n_pos": m.n_pos, "test_n_neg": m.n_neg,
            "test_auroc": m.auroc, "test_auroc_lo": m.auroc_ci[0], "test_auroc_hi": m.auroc_ci[1],
            "test_auprc": m.auprc, "test_auprc_lo": m.auprc_ci[0], "test_auprc_hi": m.auprc_ci[1],
            "test_f1": m.f1, "test_f1_lo": m.f1_ci[0], "test_f1_hi": m.f1_ci[1],
            "test_accuracy": m.accuracy,
            "test_precision": m.precision, "test_recall": m.recall, "test_specificity": m.specificity,
            "test_tn": m.confusion[0], "test_fp": m.confusion[1], "test_fn": m.confusion[2], "test_tp": m.confusion[3],
            "coverage": float(am_mask.mean()),
        })
        LOG.info("AlphaMissense test AUROC=%.4f AUPRC=%.4f (n=%d)", m.auroc, m.auprc, m.n)

    cadd_mask = joined["cadd_phred"].notna()
    if cadd_mask.any():
        y = joined.loc[cadd_mask, "label"].to_numpy(dtype=int)
        s = joined.loc[cadd_mask, "cadd_phred"].to_numpy(dtype=float)
        threshold = best_threshold_by_f1(y, s)
        m = evaluate_scores(y, s, threshold=threshold, n_boot=1000, seed=42)
        rows.append({
            "model": "cadd_phred",
            "operating_threshold": threshold,
            "test_n": m.n, "test_n_pos": m.n_pos, "test_n_neg": m.n_neg,
            "test_auroc": m.auroc, "test_auroc_lo": m.auroc_ci[0], "test_auroc_hi": m.auroc_ci[1],
            "test_auprc": m.auprc, "test_auprc_lo": m.auprc_ci[0], "test_auprc_hi": m.auprc_ci[1],
            "test_f1": m.f1, "test_f1_lo": m.f1_ci[0], "test_f1_hi": m.f1_ci[1],
            "test_accuracy": m.accuracy,
            "test_precision": m.precision, "test_recall": m.recall, "test_specificity": m.specificity,
            "test_tn": m.confusion[0], "test_fp": m.confusion[1], "test_fn": m.confusion[2], "test_tp": m.confusion[3],
            "coverage": float(cadd_mask.mean()),
        })
        LOG.info("CADD test AUROC=%.4f AUPRC=%.4f (n=%d)", m.auroc, m.auprc, m.n)

    ext_df = pd.DataFrame(rows)
    ext_path = tables_dir / "external_results.csv"
    ext_df.to_csv(ext_path, index=False)
    LOG.info("Wrote %s", ext_path)
    print(ext_df.to_string(index=False))

    main_results = pd.read_csv(tables_dir / "results.csv")
    combined = pd.concat([main_results, ext_df], ignore_index=True, sort=False)
    combined.to_csv(tables_dir / "all_results.csv", index=False)
    LOG.info("Wrote %s (%d rows)", tables_dir / "all_results.csv", len(combined))


if __name__ == "__main__":
    main()
