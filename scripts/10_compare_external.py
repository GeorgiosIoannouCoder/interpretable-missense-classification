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
from imc.eval.metrics import best_threshold_by_f1, evaluate_scores  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402

LOG = get_logger("imc.scripts.compare_external", log_file=ROOT / "logs" / "10_compare_external.log")


def main() -> None:
    """Entry point: join AlphaMissense + CADD with test set, append to results.csv."""
    cfg_data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    external_dir = ROOT / cfg_data["paths"]["external_dir"]
    tables_dir = ensure_dir(ROOT / "reports" / "tables")

    test_pred = pd.read_parquet(tables_dir / "test_predictions.parquet")
    LOG.info("Loaded %d test variants", len(test_pred))

    base_keys = test_pred[["variation_id", "chrom", "pos", "ref_nt", "alt_nt", "label", "gene"]].copy()
    base_keys["pos"] = base_keys["pos"].astype(int)
    base_keys["chrom"] = base_keys["chrom"].astype(str)

    am = load_alphamissense_for_variants(external_dir / "AlphaMissense_hg38.tsv.gz", base_keys)
    cadd = load_cadd_for_variants(external_dir / "whole_genome_SNVs.tsv.gz", base_keys)

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
