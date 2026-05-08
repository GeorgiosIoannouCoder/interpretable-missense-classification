"""Gene-disjoint train/val/test split for ClinVar missense variants.

The proposal explicitly says "our main split will hold out entire genes
rather than randomly splitting variants from the same gene across train and
test sets". This module assigns each gene to exactly one of {train, val, test}
and then propagates that assignment to its variants.

Genes are stratified by their *majority* label (P/LP vs B/LB) so neither
split is all-positive or all-negative. Within each majority-label bucket
the gene shuffle is seeded for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from imc.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass(frozen=True)
class SplitSizes:
    """Train/val/test fractions for a gene-disjoint split.

    Attributes
    ----------
    train : float
        Fraction of genes assigned to the training split.
    val : float
        Fraction of genes assigned to the validation split.
    test : float
        Fraction of genes assigned to the test split.
    """

    train: float = 0.70
    val: float = 0.10
    test: float = 0.20

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test fractions must sum to 1.0, got {total}")


def make_gene_splits(
    df: pd.DataFrame,
    sizes: SplitSizes = SplitSizes(),
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Assign each gene to a single split and propagate to variants.

    Parameters
    ----------
    df : pandas.DataFrame
        Mapped variants table (must include ``gene`` and ``label`` columns).
    sizes : SplitSizes
        Target fractions for train/val/test.
    seed : int
        Random seed for the gene shuffle.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        - The input ``df`` with an added ``split`` column in
          {"train", "val", "test"}.
        - A dict mapping each gene symbol to its split.
    """
    if "gene" not in df.columns or "label" not in df.columns:
        raise ValueError("Input dataframe must have 'gene' and 'label' columns.")

    gene_stats = (
        df.groupby("gene")["label"]
        .agg(["size", "sum"])
        .rename(columns={"size": "n", "sum": "n_pos"})
    )
    gene_stats["majority"] = (gene_stats["n_pos"] >= (gene_stats["n"] / 2.0)).astype(int)

    rng = np.random.default_rng(seed)
    gene_to_split: dict[str, str] = {}
    for majority_class in (0, 1):
        genes = gene_stats[gene_stats["majority"] == majority_class].index.to_numpy()
        rng.shuffle(genes)
        n = len(genes)
        n_train = int(round(n * sizes.train))
        n_val = int(round(n * sizes.val))
        n_test = n - n_train - n_val
        for g in genes[:n_train]:
            gene_to_split[g] = "train"
        for g in genes[n_train : n_train + n_val]:
            gene_to_split[g] = "val"
        for g in genes[n_train + n_val :]:
            gene_to_split[g] = "test"
        LOG.info(
            "majority_class=%d: %d genes -> train=%d val=%d test=%d",
            majority_class, n, n_train, n_val, n_test,
        )

    out = df.copy()
    out["split"] = out["gene"].map(gene_to_split)
    if out["split"].isna().any():
        n_missing = int(out["split"].isna().sum())
        raise RuntimeError(f"{n_missing} variants have no split assignment")
    return out, gene_to_split


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-split summary statistics for the dataset-stats table.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`make_gene_splits` (must include ``split``,
        ``label``, ``gene``, ``uniprot_acc``, ``sequence_length`` columns).

    Returns
    -------
    pandas.DataFrame
        One row per split (and a TOTAL row) with variant counts, label
        balance, gene count, protein count, and sequence-length percentiles.
    """
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test", "TOTAL"):
        sub = df if split == "TOTAL" else df[df["split"] == split]
        rows.append(
            {
                "split": split,
                "n_variants": int(len(sub)),
                "n_pos": int((sub["label"] == 1).sum()),
                "n_neg": int((sub["label"] == 0).sum()),
                "frac_pos": float((sub["label"] == 1).mean()) if len(sub) else 0.0,
                "n_genes": int(sub["gene"].nunique()),
                "n_proteins": int(sub["uniprot_acc"].nunique()),
                "median_seq_len": float(sub["sequence_length"].median()) if len(sub) else 0.0,
                "p95_seq_len": float(sub["sequence_length"].quantile(0.95)) if len(sub) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_splits(gene_to_split: dict[str, str], path: str | Path) -> None:
    """Persist the gene -> split mapping as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gene_to_split, indent=2, sort_keys=True))
