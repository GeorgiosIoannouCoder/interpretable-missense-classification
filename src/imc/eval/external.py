"""External-tool comparison: join test variants with AlphaMissense and CADD scores.

Both tools are treated as **reference points** (per the proposal) rather than
targets to outperform; we compute the same metrics on the matched
intersection so all numbers are directly comparable.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from imc.utils.logging import get_logger

LOG = get_logger(__name__)


def _normalize_chrom(c: str) -> str:
    """Normalize chromosome string to the form used by AlphaMissense (``chr1``)."""
    c = str(c)
    if c.startswith("chr"):
        return c
    return "chr" + c


def load_alphamissense_for_variants(
    am_tsv_gz: str | Path,
    test_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Filter AlphaMissense via awk -f to only rows matching the test variants.

    A pure-pandas chunked read takes ~20 minutes; an awk hashtable filter
    typically finishes in ~3 minutes. We write the awk program to a file
    (avoiding shell-quoting issues), pipe ``zcat`` into ``awk``, and load
    the resulting kilobyte-sized output with pandas.

    Parameters
    ----------
    am_tsv_gz : str or Path
        Path to ``AlphaMissense_hg38.tsv.gz``.
    test_keys : pandas.DataFrame
        Test variants with at least ``variation_id, chrom, pos, ref_nt, alt_nt``.

    Returns
    -------
    pandas.DataFrame
        Inner-join of ``test_keys`` with AlphaMissense scores, with
        ``am_pathogenicity, am_class`` columns added.
    """
    import subprocess
    import tempfile

    am_tsv_gz = Path(am_tsv_gz)
    LOG.info("Awk-filtering AlphaMissense for %d test variants ...", len(test_keys))
    keys = test_keys.copy()
    keys["chrom_am"] = keys["chrom"].map(_normalize_chrom)
    keys["pos"] = keys["pos"].astype(int)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".keys", delete=False) as f:
        for chrom_am, pos, ref, alt in zip(keys["chrom_am"], keys["pos"], keys["ref_nt"], keys["alt_nt"]):
            f.write(f"{chrom_am}\t{pos}\t{ref}\t{alt}\n")
        keys_path = f.name

    awk_path = keys_path + ".awk"
    out_path = keys_path + ".am.tsv"
    awk_script = (
        'BEGIN { FS = "\\t"; OFS = "\\t" }\n'
        'NR == FNR { seen[$1 SUBSEP $2 SUBSEP $3 SUBSEP $4] = 1; next }\n'
        '/^#/ { next }\n'
        '($1 SUBSEP $2 SUBSEP $3 SUBSEP $4) in seen { print $1, $2, $3, $4, $9, $10 }\n'
    )
    Path(awk_path).write_text(awk_script)

    cmd = f"zcat {am_tsv_gz} | awk -f {awk_path} {keys_path} - > {out_path}"
    LOG.info("Running: %s", cmd)
    subprocess.run(["bash", "-c", cmd], check=True)

    am = pd.read_csv(
        out_path,
        sep="\t",
        names=["chrom_am", "pos", "ref_nt", "alt_nt", "am_pathogenicity", "am_class"],
        dtype={"chrom_am": str, "pos": np.int64, "ref_nt": str, "alt_nt": str,
               "am_pathogenicity": np.float32, "am_class": str},
    )
    LOG.info("AlphaMissense matched %d rows", len(am))
    Path(keys_path).unlink(missing_ok=True)
    Path(awk_path).unlink(missing_ok=True)
    Path(out_path).unlink(missing_ok=True)

    out = keys.merge(am, on=["chrom_am", "pos", "ref_nt", "alt_nt"], how="left")
    return out.drop(columns=["chrom_am"])


def load_cadd_for_variants(
    cadd_tsv_gz: str | Path,
    test_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Tabix-lookup CADD PHRED scores for the test variants, one fetch per variant.

    Per-variant fetches over the bgzipped CADD file are O(1) each (single
    BGZF block lookup) and complete in well under a second per query, so
    38K test variants finish in roughly 1-2 minutes -- much faster than
    fetching whole min..max ranges per chromosome.

    Parameters
    ----------
    cadd_tsv_gz : str or Path
        Path to ``whole_genome_SNVs.tsv.gz``. The tabix index
        (``.tbi``) must be present alongside.
    test_keys : pandas.DataFrame
        Test variants with at least ``variation_id, chrom, pos, ref_nt, alt_nt``.

    Returns
    -------
    pandas.DataFrame
        ``test_keys`` with an added ``cadd_phred`` column (NaN for misses).
    """
    cadd_tsv_gz = Path(cadd_tsv_gz)
    LOG.info("Tabix-querying CADD for %d test variants (one fetch per variant) ...", len(test_keys))
    tab = pysam.TabixFile(str(cadd_tsv_gz))
    contigs = set(tab.contigs)

    keys = test_keys.copy()
    keys["pos"] = keys["pos"].astype(int)

    n = len(keys)
    out_phred = np.full(n, np.nan, dtype=np.float64)
    chrom_arr = keys["chrom"].astype(str).to_numpy()
    pos_arr = keys["pos"].to_numpy()
    ref_arr = keys["ref_nt"].astype(str).to_numpy()
    alt_arr = keys["alt_nt"].astype(str).to_numpy()

    progress_every = max(1, n // 20)
    for i in range(n):
        chrom = chrom_arr[i]
        if chrom not in contigs:
            continue
        pos = int(pos_arr[i])
        ref = ref_arr[i]
        alt = alt_arr[i]
        try:
            for line in tab.fetch(chrom, pos - 1, pos):
                cols = line.split("\t")
                if len(cols) < 6:
                    continue
                if int(cols[1]) == pos and cols[2] == ref and cols[3] == alt:
                    out_phred[i] = float(cols[5])
                    break
        except (ValueError, OSError):
            continue
        if (i + 1) % progress_every == 0:
            LOG.info(
                "CADD lookup progress: %d/%d (%.1f%%) hits=%d",
                i + 1, n, 100.0 * (i + 1) / n, int(np.isfinite(out_phred[: i + 1]).sum()),
            )

    LOG.info(
        "CADD coverage: %d / %d (%.1f%%)",
        int(np.isfinite(out_phred).sum()), n, 100.0 * np.isfinite(out_phred).mean(),
    )
    keys["cadd_phred"] = out_phred
    return keys
