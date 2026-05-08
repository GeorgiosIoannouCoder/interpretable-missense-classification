#!/usr/bin/env python3
"""Phase 5: extract handcrafted features (proposal-listed five) and persist npz.

Inputs
------
- ``data/processed/clinvar_split.parquet`` (Phase 4 output)
- ``data/raw/UP000005640_9606.fasta.gz``

Outputs
-------
- ``data/processed/features_handcrafted.npz`` with arrays:
  ``variation_id``, ``X``, ``y``, ``split``, ``feature_names``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.data.uniprot import load_swissprot_human  # noqa: E402
from imc.features.handcrafted import (  # noqa: E402
    HandcraftedConfig,
    feature_names,
    featurize,
    save_features_npz,
)
from imc.utils.io import ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402

LOG = get_logger("imc.scripts.extract_features", log_file=ROOT / "logs" / "05_extract_features.log")


def main() -> None:
    """Entry point: load splits + proteome, build features, persist npz."""
    cfg_data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    cfg_base = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text())
    raw_dir = ROOT / cfg_data["paths"]["raw_dir"]
    processed_dir = ensure_dir(ROOT / cfg_data["paths"]["processed_dir"])

    df = pd.read_parquet(processed_dir / "clinvar_split.parquet")
    LOG.info("Loaded %d variants from clinvar_split.parquet", len(df))

    swissprot = load_swissprot_human(raw_dir / "UP000005640_9606.fasta.gz")

    radius = int(cfg_base["features"]["window_radius"])
    feat_cfg = HandcraftedConfig(window_radius=radius)
    X, y = featurize(df, swissprot, cfg=feat_cfg)
    LOG.info("Built feature matrix X=%s, y=%s, dtypes=%s/%s", X.shape, y.shape, X.dtype, y.dtype)

    out_path = processed_dir / "features_handcrafted.npz"
    save_features_npz(
        out_path,
        variation_id=df["variation_id"].to_numpy().astype(str),
        X=X,
        y=y,
        splits=df["split"].to_numpy().astype(str),
        feature_names=feature_names(feat_cfg),
    )
    LOG.info("Wrote %s (size: %.1f MB)", out_path, out_path.stat().st_size / 1e6)

    pos_train = int(((y == 1) & (df["split"].to_numpy() == "train")).sum())
    pos_test = int(((y == 1) & (df["split"].to_numpy() == "test")).sum())
    LOG.info("Train pos=%d, test pos=%d", pos_train, pos_test)


if __name__ == "__main__":
    main()
