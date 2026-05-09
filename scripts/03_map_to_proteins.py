#!/usr/bin/env python3
"""Phase 3: map ClinVar missense variants to UniProt Swiss-Prot reviewed sequences.

Inputs
------
- ``data/processed/clinvar_missense.parquet`` (Phase 2 output)
- ``data/raw/UP000005640_9606.fasta.gz``
- ``data/raw/HUMAN_9606_idmapping_selected.tab.gz``

Outputs
-------
- ``data/processed/clinvar_mapped.parquet`` - variants with verified
  UniProt accession and protein sequence metadata.
- ``data/processed/mapping_funnel.csv`` - counts by mapping stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.data.mapping import load_refseq_to_uniprot, map_variants_to_uniprot  # noqa: E402
from imc.data.uniprot import load_swissprot_human  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402

LOG = get_logger("imc.scripts.map_to_proteins", log_file=ROOT / "logs" / "03_map_to_proteins.log")


def main() -> None:
    """Entry point: load proteome + idmapping, map variants, persist parquet + funnel."""
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    processed_dir = ensure_dir(ROOT / cfg["paths"]["processed_dir"])

    variants = pd.read_parquet(processed_dir / "clinvar_missense.parquet")
    LOG.info("Loaded %d variants from clinvar_missense.parquet", len(variants))

    swissprot = load_swissprot_human(raw_dir / "UP000005640_9606.fasta.gz")
    refseq_to_uniprot = load_refseq_to_uniprot(raw_dir / "HUMAN_9606_idmapping_selected.tab.gz")

    mapped, counts = map_variants_to_uniprot(variants, swissprot, refseq_to_uniprot)
    LOG.info("Mapped %d -> %d variants", counts["input_rows"], counts["mapped_rows"])

    out_parquet = processed_dir / "clinvar_mapped.parquet"
    mapped.to_parquet(out_parquet, index=False)
    LOG.info("Wrote %s", out_parquet)

    funnel = pd.DataFrame(
        [
            {"stage": "input_rows", "count": counts["input_rows"]},
            {"stage": "matched_via_refseq_versioned", "count": counts["matched_via_refseq_versioned"]},
            {"stage": "matched_via_refseq_bare", "count": counts["matched_via_refseq_bare"]},
            {"stage": "matched_via_gene_fallback", "count": counts["matched_via_gene_fallback"]},
            {"stage": "ref_aa_mismatch_dropped", "count": counts["ref_aa_mismatch_dropped"]},
            {"stage": "no_candidate_dropped", "count": counts["no_candidate_dropped"]},
            {"stage": "position_out_of_range_dropped", "count": counts["position_out_of_range_dropped"]},
            {"stage": "mapped_rows", "count": counts["mapped_rows"]},
        ]
    )
    funnel_path = processed_dir / "mapping_funnel.csv"
    funnel.to_csv(funnel_path, index=False)
    LOG.info("Wrote %s", funnel_path)


if __name__ == "__main__":
    main()
