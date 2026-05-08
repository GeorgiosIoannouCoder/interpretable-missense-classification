#!/usr/bin/env python3
"""Phase 6: extract ESM-2 (650M) residue-level embeddings, sharded and resumable.

Resumability
------------
Each shard writes one ``.npz`` and one entry in
``data/embeddings/esm2_650M/manifest.json``. Re-running the script skips
shards that are already in the manifest, so an SSH-dropped run can be
resumed simply by re-invoking the same command.

Multi-GPU
---------
Single GPU::

    python3 scripts/06_extract_esm2_embeddings.py

Multi-GPU (each rank handles a subset of shards)::

    accelerate launch --num_processes=N scripts/06_extract_esm2_embeddings.py

After all shards are present, the script consolidates them into a single
``embeddings.parquet`` keyed by ``variation_id``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.data.uniprot import load_swissprot_human  # noqa: E402
from imc.features.embeddings import ESM2Config, consolidate_shards, extract_embeddings  # noqa: E402
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402

LOG = get_logger("imc.scripts.extract_esm2", log_file=ROOT / "logs" / "06_extract_esm2_embeddings.log")


def main() -> None:
    """Entry point: load splits + proteome, run sharded extraction, consolidate."""
    cfg_data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    cfg_esm = yaml.safe_load((ROOT / "configs" / "esm2.yaml").read_text())
    raw_dir = ROOT / cfg_data["paths"]["raw_dir"]
    processed_dir = ensure_dir(ROOT / cfg_data["paths"]["processed_dir"])
    out_dir = ensure_dir(ROOT / cfg_esm["extraction"]["out_dir"])

    df = pd.read_parquet(processed_dir / "clinvar_split.parquet")
    LOG.info("Loaded %d variants from clinvar_split.parquet", len(df))

    swissprot = load_swissprot_human(raw_dir / "UP000005640_9606.fasta.gz")

    cfg = ESM2Config(
        model_id=cfg_esm["esm2"]["model_id"],
        hidden_dim=int(cfg_esm["esm2"]["hidden_dim"]),
        max_residues=int(cfg_esm["esm2"]["max_tokens"]),
        shard_size=int(cfg_esm["esm2"]["shard_size_tasks"]),
        dtype=str(cfg_esm["esm2"]["dtype"]),
        batch_size=int(cfg_esm["esm2"]["batch_size"]),
    )

    extract_embeddings(df, swissprot, cfg, out_dir=out_dir)

    consolidate_shards(out_dir)


if __name__ == "__main__":
    main()
