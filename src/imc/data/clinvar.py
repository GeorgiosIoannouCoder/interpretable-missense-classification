"""ClinVar ``variant_summary.txt.gz`` parsing and label-set construction.

Filters applied (in order, with counts logged for the report's funnel table):

1. ``Assembly == "GRCh38"``
2. Germline only (``OriginSimple`` contains ``germline``)
3. Missense: ``Type == "single nucleotide variant"`` AND ``Name`` parses to a
   p.RefAa{Position}AltAa change with two distinct standard amino acids.
4. ``ClinicalSignificance`` is one of {Pathogenic, Likely pathogenic,
   Pathogenic/Likely pathogenic} (label = 1) or {Benign, Likely benign,
   Benign/Likely benign} (label = 0). All other labels (conflicting,
   uncertain, etc.) are dropped.
5. Drop duplicates by (refseq_nm, position_aa, ref_aa, alt_aa) keeping the row
   with the strongest review status.

The output retains both the protein-level coordinates and the **genomic**
coordinates (``chrom, pos, ref_nt, alt_nt`` from ClinVar's VCF columns) so
that Phase 10's join with AlphaMissense and CADD does not require re-parsing
the source file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from imc.utils.logging import get_logger

LOG = get_logger(__name__)

# ---- amino acid alphabets ----------------------------------------------------

_THREE_TO_ONE: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
STANDARD_AAS: frozenset[str] = frozenset(_THREE_TO_ONE.values())

# ---- HGVS_p missense regex ---------------------------------------------------
# Matches things like "p.Arg175His" and rejects synonymous (p.Arg175=),
# nonsense (p.Arg175Ter / p.Arg175*), frameshift (...fs...), and extensions.
_HGVS_P_RE = re.compile(
    r"\bp\.(?P<ref>[A-Z][a-z]{2})(?P<pos>\d+)(?P<alt>[A-Z][a-z]{2})\b"
)
_NM_RE = re.compile(r"\b(NM_\d+(?:\.\d+)?)\b")
_NP_RE = re.compile(r"\b(NP_\d+(?:\.\d+)?)\b")

# ---- label sets --------------------------------------------------------------

PATHOGENIC_LABELS: frozenset[str] = frozenset({
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
})
BENIGN_LABELS: frozenset[str] = frozenset({
    "Benign",
    "Likely benign",
    "Benign/Likely benign",
})

# ---- review-status -> star rating (per NCBI's ClinVar docs) ------------------

REVIEW_STATUS_STARS: dict[str, int] = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,  # legacy phrasing
    "criteria provided, single submitter": 1,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
    "no classifications from unflagged records": 0,
    "no assertion provided": 0,
}


@dataclass(frozen=True)
class ParsedMissense:
    """Parsed missense components from a ClinVar HGVS ``Name``.

    Attributes
    ----------
    refseq_nm : str
        RefSeq mRNA / transcript accession, e.g. ``NM_007294.4``.
    refseq_np : str or None
        RefSeq protein accession (``NP_xxx``) if explicitly present in
        ``Name``; otherwise ``None`` (Phase 3 will resolve via id-mapping).
    position_aa : int
        1-based amino-acid position of the variant.
    ref_aa : str
        Reference amino acid, one-letter code.
    alt_aa : str
        Alternate amino acid, one-letter code.
    """

    refseq_nm: str
    refseq_np: str | None
    position_aa: int
    ref_aa: str
    alt_aa: str


def parse_missense_name(name: str) -> ParsedMissense | None:
    """Parse a ClinVar ``Name`` string for a missense substitution.

    Returns ``None`` if the row is not a clean missense (synonymous,
    nonsense, frameshift, malformed, or non-standard amino acid).

    Parameters
    ----------
    name : str
        Raw ClinVar ``Name`` field.

    Returns
    -------
    ParsedMissense or None
        Parsed components, or ``None`` if the row is not a clean missense.
    """
    if not isinstance(name, str) or "p." not in name:
        return None
    m = _HGVS_P_RE.search(name)
    if not m:
        return None
    ref3 = m.group("ref")
    alt3 = m.group("alt")
    pos = int(m.group("pos"))
    ref1 = _THREE_TO_ONE.get(ref3)
    alt1 = _THREE_TO_ONE.get(alt3)
    if ref1 is None or alt1 is None:
        return None
    if ref1 == alt1:  # synonymous / extension
        return None
    if ref1 not in STANDARD_AAS or alt1 not in STANDARD_AAS:
        return None
    nm_m = _NM_RE.search(name)
    np_m = _NP_RE.search(name)
    if nm_m is None:
        return None
    return ParsedMissense(
        refseq_nm=nm_m.group(1),
        refseq_np=np_m.group(1) if np_m else None,
        position_aa=pos,
        ref_aa=ref1,
        alt_aa=alt1,
    )


def review_status_to_stars(status: str) -> int:
    """Map a ClinVar ``ReviewStatus`` string to its star rating (0-4).

    Parameters
    ----------
    status : str
        Raw ``ReviewStatus`` value from ClinVar.

    Returns
    -------
    int
        Star rating in {0, 1, 2, 3, 4}. Unknown statuses default to 0.
    """
    if not isinstance(status, str):
        return 0
    return REVIEW_STATUS_STARS.get(status.strip().lower(), 0)


def filter_clinvar(input_path: str | Path, *, chunksize: int = 200_000) -> tuple[pd.DataFrame, dict[str, int]]:
    """Stream ``variant_summary.txt.gz`` and return filtered missense P/LP-vs-B/LB rows.

    Parameters
    ----------
    input_path : str or Path
        Path to the gzipped ClinVar variant summary file.
    chunksize : int
        Number of rows per pandas read chunk.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        The filtered dataframe (with `variation_id, gene, chrom, pos, ref_nt,
        alt_nt, refseq_nm, refseq_np, hgvs_p, position_aa, ref_aa, alt_aa,
        label, review_status, review_stars, clinical_significance` columns),
        and a dict of stage counts for the report's funnel table.
    """
    input_path = Path(input_path)
    LOG.info("Reading ClinVar variant_summary from %s", input_path)

    use_cols = [
        "#AlleleID", "Type", "Name", "GeneSymbol",
        "ClinicalSignificance", "OriginSimple", "Assembly",
        "Chromosome", "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF",
        "ReviewStatus", "VariationID",
    ]
    counts: dict[str, int] = {
        "raw_rows": 0,
        "after_grch38": 0,
        "after_germline": 0,
        "after_snv": 0,
        "after_missense_parse": 0,
        "after_label_filter": 0,
    }
    frames: list[pd.DataFrame] = []

    reader = pd.read_csv(
        input_path,
        sep="\t",
        compression="gzip",
        usecols=use_cols,
        dtype=str,
        na_filter=False,
        low_memory=False,
        chunksize=chunksize,
    )
    for chunk in reader:
        counts["raw_rows"] += len(chunk)

        chunk = chunk[chunk["Assembly"] == "GRCh38"]
        counts["after_grch38"] += len(chunk)

        chunk = chunk[chunk["OriginSimple"].str.contains("germline", case=False, regex=False, na=False)]
        counts["after_germline"] += len(chunk)

        chunk = chunk[chunk["Type"] == "single nucleotide variant"]
        counts["after_snv"] += len(chunk)

        parsed = chunk["Name"].map(parse_missense_name)
        keep = parsed.notna()
        chunk = chunk.loc[keep].copy()
        parsed = parsed.loc[keep]
        chunk["refseq_nm"] = parsed.map(lambda p: p.refseq_nm)
        chunk["refseq_np"] = parsed.map(lambda p: p.refseq_np)
        chunk["position_aa"] = parsed.map(lambda p: p.position_aa).astype("int64")
        chunk["ref_aa"] = parsed.map(lambda p: p.ref_aa)
        chunk["alt_aa"] = parsed.map(lambda p: p.alt_aa)
        counts["after_missense_parse"] += len(chunk)

        sig = chunk["ClinicalSignificance"]
        is_path = sig.isin(PATHOGENIC_LABELS)
        is_ben = sig.isin(BENIGN_LABELS)
        chunk = chunk[is_path | is_ben].copy()
        chunk["label"] = chunk["ClinicalSignificance"].isin(PATHOGENIC_LABELS).astype("int8")
        counts["after_label_filter"] += len(chunk)

        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        raise RuntimeError("No rows survived filtering; check ClinVar file integrity and column names.")

    df = pd.concat(frames, ignore_index=True)
    df["review_status"] = df["ReviewStatus"]
    df["review_stars"] = df["ReviewStatus"].map(review_status_to_stars).astype("int8")

    df = df.rename(
        columns={
            "VariationID": "variation_id",
            "GeneSymbol": "gene",
            "Chromosome": "chrom",
            "PositionVCF": "pos",
            "ReferenceAlleleVCF": "ref_nt",
            "AlternateAlleleVCF": "alt_nt",
            "Name": "hgvs_p",
            "ClinicalSignificance": "clinical_significance",
        }
    )
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    df = df[df["pos"].notna()].copy()
    df["pos"] = df["pos"].astype("int64")

    cols = [
        "variation_id", "gene", "chrom", "pos", "ref_nt", "alt_nt",
        "refseq_nm", "refseq_np", "hgvs_p", "position_aa", "ref_aa", "alt_aa",
        "label", "review_status", "review_stars", "clinical_significance",
    ]
    df = df[cols]

    before_dedup = len(df)
    df = (
        df.sort_values(["review_stars", "variation_id"], ascending=[False, True])
        .drop_duplicates(subset=["refseq_nm", "position_aa", "ref_aa", "alt_aa"], keep="first")
        .reset_index(drop=True)
    )
    counts["after_dedup"] = len(df)
    LOG.info("Dedup: %d -> %d rows by (refseq_nm, position_aa, ref_aa, alt_aa)", before_dedup, len(df))

    counts["pos_label"] = int((df["label"] == 1).sum())
    counts["neg_label"] = int((df["label"] == 0).sum())
    counts["unique_genes"] = df["gene"].nunique()
    counts["unique_refseq_nm"] = df["refseq_nm"].nunique()

    return df, counts
