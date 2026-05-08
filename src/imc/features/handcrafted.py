"""Handcrafted sequence-based features for the interpretable baseline.

The headline baseline uses **exactly** the five feature families listed in
the approved proposal:

1. One-hot reference amino acid (20-d).
2. One-hot alternate amino acid (20-d).
3. BLOSUM62 substitution score for (ref, alt) (1-d).
4. Local sequence-window amino-acid composition centered on the variant
   position (window radius :math:`w`; 20-d frequency vector).
5. Normalized residue position = ``position_aa / sequence_length`` (1-d).

Total feature dimensionality: ``20 + 20 + 1 + 20 + 1 = 62`` (with default
``w=7``). The dimensionality of the window-composition block is independent
of ``w`` because it is a fixed 20-d frequency vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

from imc.data.uniprot import UniprotEntry
from imc.utils.logging import get_logger

LOG = get_logger(__name__)

AA_ALPHABET: tuple[str, ...] = (
    "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
)
AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AA_ALPHABET)}


def _build_blosum62_matrix() -> np.ndarray:
    """Materialize a 20x20 NumPy matrix of BLOSUM62 substitution scores."""
    bl = substitution_matrices.load("BLOSUM62")
    mat = np.zeros((20, 20), dtype=np.float32)
    for i, a in enumerate(AA_ALPHABET):
        for j, b in enumerate(AA_ALPHABET):
            mat[i, j] = float(bl[a, b])
    return mat


BLOSUM62: np.ndarray = _build_blosum62_matrix()


@dataclass(frozen=True)
class HandcraftedConfig:
    """Hyperparameters for handcrafted feature extraction.

    Attributes
    ----------
    window_radius : int
        Half-width of the local sequence window (residues on each side).
        Default 7 → 14 residues of context around the variant residue.
    """

    window_radius: int = 7


def feature_dim(_: HandcraftedConfig | None = None) -> int:
    """Return the dimensionality of the headline handcrafted feature vector.

    The dimensionality is independent of the window radius because the
    composition block is always a 20-d frequency vector.

    Parameters
    ----------
    _ : HandcraftedConfig or None
        Ignored; present for signature symmetry with other feature builders.

    Returns
    -------
    int
        Total feature vector length (62 with defaults).
    """
    return 20 + 20 + 1 + 20 + 1


def feature_names(_: HandcraftedConfig | None = None) -> list[str]:
    """Return ordered feature names matching :func:`featurize`.

    Returns
    -------
    list[str]
        Column names in the order produced by :func:`featurize`. Used for
        feature-importance plots in the appendix.
    """
    names = [f"ref_aa_{aa}" for aa in AA_ALPHABET]
    names += [f"alt_aa_{aa}" for aa in AA_ALPHABET]
    names += ["blosum62"]
    names += [f"window_freq_{aa}" for aa in AA_ALPHABET]
    names += ["norm_position"]
    return names


def _encode_window(seq: str, position_aa: int, radius: int) -> np.ndarray:
    """Build a 20-d AA-frequency vector for a window centered on the variant.

    Parameters
    ----------
    seq : str
        Full protein sequence (one-letter AA codes).
    position_aa : int
        1-based variant position.
    radius : int
        Number of residues on each side of the variant residue to include.

    Returns
    -------
    numpy.ndarray
        Frequency vector of shape ``(20,)`` summing to 1.0 (or 0.0 if the
        window happens to contain only non-standard residues).
    """
    n = len(seq)
    lo = max(0, position_aa - 1 - radius)
    hi = min(n, position_aa - 1 + radius + 1)
    window = seq[lo:hi]
    counts = np.zeros(20, dtype=np.float32)
    if not window:
        return counts
    for residue in window:
        idx = AA_INDEX.get(residue)
        if idx is not None:
            counts[idx] += 1.0
    total = counts.sum()
    if total > 0.0:
        counts /= total
    return counts


def featurize(
    variants: pd.DataFrame,
    swissprot: dict[str, UniprotEntry],
    cfg: HandcraftedConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the headline handcrafted feature matrix for ``variants``.

    Parameters
    ----------
    variants : pandas.DataFrame
        Mapped variants with at least ``uniprot_acc, position_aa, ref_aa,
        alt_aa, sequence_length`` columns.
    swissprot : dict[str, UniprotEntry]
        Output of :func:`imc.data.uniprot.load_swissprot_human` keyed by
        UniProt accession.
    cfg : HandcraftedConfig or None
        Feature config; defaults to ``HandcraftedConfig()``.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        - Feature matrix of shape ``(n_variants, feature_dim())`` and dtype
          ``float32``.
        - Labels array of shape ``(n_variants,)`` and dtype ``int8`` aligned
          with the matrix.
    """
    cfg = cfg or HandcraftedConfig()
    n = len(variants)
    d = feature_dim(cfg)
    X = np.zeros((n, d), dtype=np.float32)

    accs = variants["uniprot_acc"].to_numpy()
    pos = variants["position_aa"].to_numpy()
    ref_aas = variants["ref_aa"].to_numpy()
    alt_aas = variants["alt_aa"].to_numpy()
    seq_lens = variants["sequence_length"].to_numpy()
    labels = variants["label"].to_numpy().astype(np.int8)

    LOG.info("Featurizing %d variants (dim=%d, window_radius=%d)", n, d, cfg.window_radius)
    for i in range(n):
        ref = ref_aas[i]
        alt = alt_aas[i]
        ref_idx = AA_INDEX[ref]
        alt_idx = AA_INDEX[alt]

        # 1) one-hot ref AA (cols 0..19)
        X[i, ref_idx] = 1.0
        # 2) one-hot alt AA (cols 20..39)
        X[i, 20 + alt_idx] = 1.0
        # 3) BLOSUM62(ref, alt) (col 40)
        X[i, 40] = BLOSUM62[ref_idx, alt_idx]
        # 4) window composition (cols 41..60)
        entry = swissprot.get(accs[i])
        if entry is not None:
            X[i, 41:61] = _encode_window(entry.sequence, int(pos[i]), cfg.window_radius)
        # 5) normalized position (col 61)
        sl = float(seq_lens[i]) if seq_lens[i] else 1.0
        X[i, 61] = float(pos[i]) / sl if sl > 0 else 0.0

    return X, labels


def save_features_npz(
    path: str | Path,
    *,
    variation_id: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save feature matrix + aligned arrays to a single ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        variation_id=variation_id,
        X=X,
        y=y,
        split=splits,
        feature_names=np.array(feature_names, dtype=object),
    )
