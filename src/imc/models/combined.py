"""Classifier head over concatenated handcrafted sequence features and ESM-2.

The architecture matches :class:`~imc.models.head.ESM2HeadMLP` but the default
input dimension is ``62 + 1280 = 1342`` (proposal handcrafted features plus
ESM-2 650M residue embedding).
"""

from __future__ import annotations

from imc.models.head import ESM2HeadMLP


class CombinedHead(ESM2HeadMLP):
    """Small MLP on **concatenated** handcrafted + frozen ESM-2 vectors.

    Parameters are identical to :class:`~imc.models.head.ESM2HeadMLP`; only
    ``in_dim`` is expected to be ``1342`` when using the headline feature set.
    """
