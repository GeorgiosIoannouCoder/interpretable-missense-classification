"""UniProt UP000005640 (human reference proteome) loader, restricted to Swiss-Prot reviewed entries.

The proposal explicitly says we map to a "reviewed human protein sequence",
so this loader filters to ``>sp|`` headers and drops TrEMBL (``>tr|``).
"""

from __future__ import annotations

import gzip
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from imc.utils.logging import get_logger

LOG = get_logger(__name__)

_GN_RE = re.compile(r"\bGN=(\S+)")


@dataclass(frozen=True)
class UniprotEntry:
    """A single Swiss-Prot reviewed UniProt entry from the human proteome.

    Attributes
    ----------
    accession : str
        UniProtKB accession (canonical only; no ``-N`` isoform suffixes).
    gene : str or None
        ``GN=`` gene symbol from the FASTA header, if present.
    sequence : str
        Amino-acid sequence (one-letter codes).
    """

    accession: str
    gene: str | None
    sequence: str

    @property
    def length(self) -> int:
        """Sequence length."""
        return len(self.sequence)

    @property
    def sha1(self) -> str:
        """SHA-1 of the amino-acid sequence (for change tracking)."""
        return hashlib.sha1(self.sequence.encode("ascii")).hexdigest()


def load_swissprot_human(fasta_gz_path: str | Path) -> dict[str, UniprotEntry]:
    """Load all Swiss-Prot reviewed entries from UP000005640.

    Parameters
    ----------
    fasta_gz_path : str or Path
        Path to ``UP000005640_9606.fasta.gz``.

    Returns
    -------
    dict[str, UniprotEntry]
        Mapping from UniProt accession to ``UniprotEntry``.
    """
    fasta_gz_path = Path(fasta_gz_path)
    LOG.info("Loading UP000005640 Swiss-Prot entries from %s", fasta_gz_path)
    entries: dict[str, UniprotEntry] = {}
    skipped_tremble = 0
    cur_acc: str | None = None
    cur_gene: str | None = None
    cur_seq_chunks: list[str] = []

    def flush() -> None:
        if cur_acc is not None:
            entries[cur_acc] = UniprotEntry(
                accession=cur_acc,
                gene=cur_gene,
                sequence="".join(cur_seq_chunks),
            )

    with gzip.open(fasta_gz_path, "rt", encoding="ascii") as f:
        for line in f:
            if line.startswith(">"):
                flush()
                cur_seq_chunks = []
                if not line.startswith(">sp|"):
                    cur_acc = None
                    skipped_tremble += 1
                    continue
                parts = line[1:].rstrip().split("|", 2)
                if len(parts) < 3:
                    cur_acc = None
                    continue
                cur_acc = parts[1]
                gn_m = _GN_RE.search(parts[2])
                cur_gene = gn_m.group(1) if gn_m else None
            else:
                if cur_acc is not None:
                    cur_seq_chunks.append(line.strip())
        flush()

    LOG.info(
        "Loaded %d Swiss-Prot entries (%d TrEMBL entries skipped) from %s",
        len(entries), skipped_tremble, fasta_gz_path.name,
    )
    return entries


def gene_to_canonical_index(entries: dict[str, UniprotEntry]) -> dict[str, str]:
    """Build a gene-symbol -> UniProt accession lookup.

    If a gene symbol maps to multiple accessions (rare for canonical-only
    Swiss-Prot), the longest sequence wins as a heuristic for "canonical".

    Parameters
    ----------
    entries : dict[str, UniprotEntry]
        Output of :func:`load_swissprot_human`.

    Returns
    -------
    dict[str, str]
        Mapping from gene symbol to canonical UniProt accession.
    """
    index: dict[str, str] = {}
    lengths: dict[str, int] = {}
    for acc, e in entries.items():
        if e.gene is None:
            continue
        if e.gene not in index or e.length > lengths[e.gene]:
            index[e.gene] = acc
            lengths[e.gene] = e.length
    return index
