#!/usr/bin/env python3
"""Phase 2: filter ClinVar to germline missense P/LP vs B/LB and persist parquet.

Outputs
-------
- ``data/processed/clinvar_missense.parquet`` - filtered + deduped variants.
- ``data/processed/clinvar_funnel.csv`` - per-stage retention counts (Table 1 source).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.data.clinvar import filter_clinvar  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402

LOG = get_logger("imc.scripts.preprocess_clinvar", log_file=ROOT / "logs" / "02_preprocess_clinvar.log")


def main() -> None:
    """Entry point: filter ClinVar and persist parquet + funnel CSV."""
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    processed_dir = ensure_dir(ROOT / cfg["paths"]["processed_dir"])
    input_path = raw_dir / cfg["clinvar"]["filename"]

    df, counts = filter_clinvar(input_path)

    out_parquet = processed_dir / "clinvar_missense.parquet"
    df.to_parquet(out_parquet, index=False)
    LOG.info("Wrote %s (%d rows, %d cols)", out_parquet, len(df), df.shape[1])

    funnel = pd.DataFrame(
        [
            {"stage": "raw_rows", "count": counts["raw_rows"]},
            {"stage": "after_assembly_grch38", "count": counts["after_grch38"]},
            {"stage": "after_germline", "count": counts["after_germline"]},
            {"stage": "after_snv_type", "count": counts["after_snv"]},
            {"stage": "after_missense_parse", "count": counts["after_missense_parse"]},
            {"stage": "after_label_path_or_benign", "count": counts["after_label_filter"]},
            {"stage": "after_dedup", "count": counts["after_dedup"]},
        ]
    )
    funnel_path = processed_dir / "clinvar_funnel.csv"
    funnel.to_csv(funnel_path, index=False)
    LOG.info("Wrote funnel: %s", funnel_path)

    summary = {
        "rows": int(len(df)),
        "pos_label": counts["pos_label"],
        "neg_label": counts["neg_label"],
        "unique_genes": counts["unique_genes"],
        "unique_refseq_nm": counts["unique_refseq_nm"],
        "stars_distribution": (
            df["review_stars"].value_counts().sort_index().to_dict()
        ),
    }
    summary_path = processed_dir / "clinvar_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote summary: %s", summary_path)

    LOG.info("Funnel: %s", funnel.to_dict("records"))
    LOG.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
