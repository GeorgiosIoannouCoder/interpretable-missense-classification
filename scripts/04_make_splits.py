#!/usr/bin/env python3
"""Phase 4: assign each gene to train/val/test and write per-split stats.

Inputs
------
- ``data/processed/clinvar_mapped.parquet`` (Phase 3 output)

Outputs
-------
- ``data/processed/clinvar_split.parquet`` - variants tagged with ``split``.
- ``data/processed/splits.json`` - gene -> split mapping.
- ``reports/tables/dataset_stats.csv`` - Table 1 source (per-split stats).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.data.splits import SplitSizes, make_gene_splits, split_summary, write_splits  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402

LOG = get_logger("imc.scripts.make_splits", log_file=ROOT / "logs" / "04_make_splits.log")


def main() -> None:
    """Entry point: assign splits, persist parquet + splits.json + dataset_stats.csv."""
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    processed_dir = ensure_dir(ROOT / cfg["paths"]["processed_dir"])
    tables_dir = ensure_dir(ROOT / "reports" / "tables")

    df = pd.read_parquet(processed_dir / "clinvar_mapped.parquet")
    LOG.info("Loaded %d mapped variants", len(df))

    sizes = SplitSizes(
        train=cfg["splits"]["train_frac"],
        val=cfg["splits"]["val_frac"],
        test=cfg["splits"]["test_frac"],
    )
    df_split, gene_to_split = make_gene_splits(df, sizes=sizes, seed=cfg["seed"])

    out_parquet = processed_dir / "clinvar_split.parquet"
    df_split.to_parquet(out_parquet, index=False)
    LOG.info("Wrote %s", out_parquet)

    write_splits(gene_to_split, processed_dir / "splits.json")
    LOG.info("Wrote splits.json (%d genes)", len(gene_to_split))

    summary = split_summary(df_split)
    summary_path = tables_dir / "dataset_stats.csv"
    summary.to_csv(summary_path, index=False)
    LOG.info("Wrote %s", summary_path)
    LOG.info("Summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
