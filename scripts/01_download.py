#!/usr/bin/env python3
"""Phase 1: download all source datasets to ``data/raw/`` and ``data/external/``.

Usage
-----
Foreground (everything except the 87 GB CADD bulk file)::

    python3 scripts/01_download.py

CADD bulk file only (run as a background ``nohup`` job at the start of Day 1
so it streams while the rest of the pipeline executes)::

    nohup python3 scripts/01_download.py --cadd-only > logs/cadd_download.log 2>&1 &

The script is idempotent: completed files are kept; partial files
(``*.part``) are resumed via HTTP Range requests.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imc.utils.io import download_file, ensure_dir  # noqa: E402
from imc.utils.logging import get_logger  # noqa: E402

LOG = get_logger("imc.scripts.download", log_file=ROOT / "logs" / "01_download.log")


def _record_clinvar_release(url: str, dest: Path) -> None:
    """Record the ClinVar FTP ``Last-Modified`` date so Methods can cite it.

    Parameters
    ----------
    url : str
        ClinVar variant_summary URL.
    dest : Path
        Path to write the release-date marker file.
    """
    try:
        head = requests.head(url, timeout=30, allow_redirects=True)
        head.raise_for_status()
        last_modified = head.headers.get("Last-Modified", "unknown")
    except requests.RequestException as exc:
        LOG.warning("Could not read ClinVar Last-Modified header: %s", exc)
        last_modified = "unknown"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        f"clinvar_url: {url}\n"
        f"last_modified: {last_modified}\n"
        f"recorded_at: {datetime.utcnow().isoformat()}Z\n"
    )
    LOG.info("Recorded ClinVar release: Last-Modified=%s -> %s", last_modified, dest)


def _download_cadd(cfg: dict, raw_dir: Path) -> None:
    """Download the CADD bulk file (~87 GB) and tabix index.

    Parameters
    ----------
    cfg : dict
        Loaded ``configs/data.yaml`` mapping.
    raw_dir : Path
        Destination directory for downloaded files.
    """
    LOG.info("Downloading CADD .tbi index ...")
    download_file(
        url=cfg["cadd"]["index_url"],
        dest=raw_dir / "whole_genome_SNVs.tsv.gz.tbi",
    )
    LOG.info("Downloading CADD bulk file (~87 GB; this is the slow one) ...")
    download_file(
        url=cfg["cadd"]["scores_url"],
        dest=raw_dir / "whole_genome_SNVs.tsv.gz",
    )
    LOG.info("CADD download complete.")


def _download_non_cadd(cfg: dict, raw_dir: Path, external_dir: Path, processed_dir: Path) -> None:
    """Download every source except the CADD bulk file.

    Parameters
    ----------
    cfg : dict
        Loaded ``configs/data.yaml`` mapping.
    raw_dir : Path
        Destination directory for ClinVar / UniProt files.
    external_dir : Path
        Destination directory for AlphaMissense / CADD index.
    processed_dir : Path
        Destination directory for the ClinVar release marker file.
    """
    LOG.info("Downloading ClinVar variant_summary.txt.gz ...")
    download_file(
        url=cfg["clinvar"]["url"],
        dest=raw_dir / cfg["clinvar"]["filename"],
    )
    _record_clinvar_release(
        url=cfg["clinvar"]["url"],
        dest=processed_dir / "clinvar_release.txt",
    )

    LOG.info("Downloading UniProt UP000005640 FASTA ...")
    download_file(
        url=cfg["uniprot"]["proteome_url"],
        dest=raw_dir / "UP000005640_9606.fasta.gz",
    )

    LOG.info("Downloading UniProt RefSeq->UniProt id mapping ...")
    download_file(
        url=cfg["uniprot"]["idmapping_url"],
        dest=raw_dir / "HUMAN_9606_idmapping_selected.tab.gz",
    )

    LOG.info("Downloading AlphaMissense scores ...")
    download_file(
        url=cfg["alphamissense"]["url"],
        dest=external_dir / cfg["alphamissense"]["filename"],
    )

    LOG.info("Downloading CADD .tbi tabix index (small) ...")
    download_file(
        url=cfg["cadd"]["index_url"],
        dest=external_dir / "whole_genome_SNVs.tsv.gz.tbi",
    )


def main() -> None:
    """Entry point: parse args, load config, dispatch downloads."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "data.yaml",
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--cadd-only",
        action="store_true",
        help="Only download the CADD bulk file (~87 GB) and its .tbi. Use as a nohup background job.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    raw_dir = ensure_dir(ROOT / cfg["paths"]["raw_dir"])
    external_dir = ensure_dir(ROOT / cfg["paths"]["external_dir"])
    processed_dir = ensure_dir(ROOT / cfg["paths"]["processed_dir"])

    if args.cadd_only:
        _download_cadd(cfg, raw_dir=external_dir)
        return

    _download_non_cadd(cfg, raw_dir=raw_dir, external_dir=external_dir, processed_dir=processed_dir)
    LOG.info("Foreground downloads complete. Run --cadd-only as a background nohup job for the 87 GB CADD bulk file.")


if __name__ == "__main__":
    main()
