"""Map ClinVar missense variants to UniProt Swiss-Prot reviewed sequences.

The mapping strategy (in order of preference) is:

1. Exact RefSeq mRNA accession with version (e.g. ``NM_007294.4``) in the
   UniProt RefSeq id-mapping file.
2. RefSeq mRNA accession **without** version (drop ``.4`` suffix).
3. Gene-symbol fallback against UP000005640 canonical entries.

Every candidate UniProt accession is then verified by checking that the
amino acid at ``position_aa`` of the UniProt sequence equals the HGVS
``ref_aa`` reported by ClinVar. Mismatches are dropped and counted (the
count goes into the report's funnel table).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from imc.data.uniprot import UniprotEntry, gene_to_canonical_index, load_swissprot_human
from imc.utils.logging import get_logger

LOG = get_logger(__name__)


def _strip_version(acc: str) -> str:
    """Drop the ``.<n>`` version suffix from a RefSeq accession."""
    return acc.split(".", 1)[0]


def load_refseq_to_uniprot(idmapping_gz_path: str | Path) -> dict[str, list[str]]:
    """Build a RefSeq accession -> list of UniProt accessions index.

    The UniProt id-mapping file is tab-separated; column 1 is the UniProt
    accession and column 4 is a semicolon-separated list of RefSeq IDs
    (both ``NM_...`` and ``NP_...``). We index by both versioned and
    unversioned RefSeq IDs.

    Parameters
    ----------
    idmapping_gz_path : str or Path
        Path to ``HUMAN_9606_idmapping_selected.tab.gz``.

    Returns
    -------
    dict[str, list[str]]
        Mapping from RefSeq accession (with and without version) to a
        list of candidate UniProt accessions.
    """
    idmapping_gz_path = Path(idmapping_gz_path)
    LOG.info("Loading RefSeq -> UniProt id mapping from %s", idmapping_gz_path)
    index: dict[str, list[str]] = {}
    n_rows = 0
    with gzip.open(idmapping_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            n_rows += 1
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                continue
            uniprot_acc = cols[0]
            refseq_field = cols[3]
            if not refseq_field:
                continue
            for raw in refseq_field.split(";"):
                ref = raw.strip()
                if not ref:
                    continue
                index.setdefault(ref, []).append(uniprot_acc)
                bare = _strip_version(ref)
                if bare != ref:
                    index.setdefault(bare, []).append(uniprot_acc)
    LOG.info("idmapping: %d rows scanned, %d distinct RefSeq keys indexed", n_rows, len(index))
    return index


def _candidate_accessions(
    refseq_nm: str,
    gene: str | None,
    refseq_to_uniprot: dict[str, list[str]],
    gene_index: dict[str, str],
) -> list[str]:
    """Return ordered list of UniProt accessions to try for a variant.

    Parameters
    ----------
    refseq_nm : str
        RefSeq mRNA accession from ClinVar (with version).
    gene : str or None
        Gene symbol from ClinVar, used only as a fallback.
    refseq_to_uniprot : dict[str, list[str]]
        Output of :func:`load_refseq_to_uniprot`.
    gene_index : dict[str, str]
        Output of :func:`imc.data.uniprot.gene_to_canonical_index`.

    Returns
    -------
    list[str]
        Ordered, deduplicated UniProt accessions.
    """
    seen: set[str] = set()
    out: list[str] = []
    for key in (refseq_nm, _strip_version(refseq_nm)):
        for acc in refseq_to_uniprot.get(key, []):
            if acc not in seen:
                seen.add(acc)
                out.append(acc)
    if gene and gene in gene_index:
        acc = gene_index[gene]
        if acc not in seen:
            seen.add(acc)
            out.append(acc)
    return out


def map_variants_to_uniprot(
    variants: pd.DataFrame,
    swissprot: dict[str, UniprotEntry],
    refseq_to_uniprot: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach a verified UniProt accession to each ClinVar missense variant.

    Parameters
    ----------
    variants : pandas.DataFrame
        Output of :func:`imc.data.clinvar.filter_clinvar`.
    swissprot : dict[str, UniprotEntry]
        Output of :func:`imc.data.uniprot.load_swissprot_human`.
    refseq_to_uniprot : dict[str, list[str]]
        Output of :func:`load_refseq_to_uniprot`.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        The mapped variants (with ``uniprot_acc, sequence_length,
        sequence_sha1`` columns) and a dict of per-stage counts.
    """
    LOG.info("Mapping %d variants to UniProt Swiss-Prot ...", len(variants))
    gene_index = gene_to_canonical_index(swissprot)

    n = len(variants)
    uniprot_acc_arr = [None] * n
    seq_len_arr: list[int] = [0] * n
    seq_sha1_arr: list[str] = [""] * n
    matched_via_arr: list[str] = ["unmatched"] * n

    counts = {
        "input_rows": n,
        "matched_via_refseq_versioned": 0,
        "matched_via_refseq_bare": 0,
        "matched_via_gene_fallback": 0,
        "ref_aa_mismatch_dropped": 0,
        "no_candidate_dropped": 0,
        "position_out_of_range_dropped": 0,
    }

    refseq_nm_arr = variants["refseq_nm"].to_numpy()
    gene_arr = variants["gene"].to_numpy()
    pos_arr = variants["position_aa"].to_numpy()
    ref_aa_arr = variants["ref_aa"].to_numpy()

    for i in range(n):
        nm = refseq_nm_arr[i]
        gene = gene_arr[i]
        pos = int(pos_arr[i])
        ref_aa = ref_aa_arr[i]

        candidates = _candidate_accessions(nm, gene, refseq_to_uniprot, gene_index)
        if not candidates:
            counts["no_candidate_dropped"] += 1
            continue

        winner = None
        winner_via = None
        for acc in candidates:
            entry = swissprot.get(acc)
            if entry is None:
                continue
            if pos < 1 or pos > entry.length:
                continue
            if entry.sequence[pos - 1] == ref_aa:
                winner = entry
                if acc in refseq_to_uniprot.get(nm, []):
                    winner_via = "refseq_versioned"
                elif acc in refseq_to_uniprot.get(_strip_version(nm), []):
                    winner_via = "refseq_bare"
                else:
                    winner_via = "gene_fallback"
                break

        if winner is None:
            any_candidate_in_range = any(
                (acc in swissprot and 1 <= pos <= swissprot[acc].length) for acc in candidates
            )
            if any_candidate_in_range:
                counts["ref_aa_mismatch_dropped"] += 1
            else:
                counts["position_out_of_range_dropped"] += 1
            continue

        uniprot_acc_arr[i] = winner.accession
        seq_len_arr[i] = winner.length
        seq_sha1_arr[i] = winner.sha1
        matched_via_arr[i] = winner_via
        counts[f"matched_via_{winner_via}"] += 1

    out = variants.copy()
    out["uniprot_acc"] = uniprot_acc_arr
    out["sequence_length"] = seq_len_arr
    out["sequence_sha1"] = seq_sha1_arr
    out["matched_via"] = matched_via_arr

    keep = out["uniprot_acc"].notna()
    out = out.loc[keep].reset_index(drop=True)
    counts["mapped_rows"] = int(len(out))

    out = out[
        [
            "variation_id", "gene", "chrom", "pos", "ref_nt", "alt_nt",
            "refseq_nm", "refseq_np", "uniprot_acc", "hgvs_p",
            "position_aa", "ref_aa", "alt_aa",
            "sequence_length", "sequence_sha1",
            "label", "review_status", "review_stars", "clinical_significance",
            "matched_via",
        ]
    ]
    LOG.info("Mapping counts: %s", counts)
    return out, counts
