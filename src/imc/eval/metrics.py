"""Evaluation metrics with bootstrap 95% confidence intervals.

Metrics implemented: AUROC, AUPRC, F1 (operating point chosen on val),
accuracy, precision, recall, specificity. Confidence intervals are computed
via the percentile bootstrap over test variants (default 1000 resamples).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricsResult:
    """Container for a model's metrics on a single split."""

    n: int
    n_pos: int
    n_neg: int
    auroc: float
    auprc: float
    f1: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    threshold: float
    auroc_ci: tuple[float, float]
    auprc_ci: tuple[float, float]
    f1_ci: tuple[float, float]
    confusion: tuple[int, int, int, int]


def _confusion_components(tn: int, fp: int, fn: int, tp: int) -> tuple[float, float, float, float]:
    """Compute precision, recall, specificity, accuracy from confusion entries."""
    n = tn + fp + fn + tp
    accuracy = float((tp + tn) / max(1, n))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    specificity = float(tn / max(1, tn + fp))
    return precision, recall, specificity, accuracy


def best_threshold_by_f1(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """Find the operating-point threshold that maximizes F1 on the validation split.

    Parameters
    ----------
    y_val : numpy.ndarray
        Validation labels (0/1).
    scores_val : numpy.ndarray
        Validation scores in [0, 1].

    Returns
    -------
    float
        Threshold ``t`` such that predicted positive iff ``score >= t``.
    """
    candidates = np.linspace(0.0, 1.0, 201)
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        preds = (scores_val >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def _bootstrap_ci(
    metric_fn,
    y: np.ndarray,
    s: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile-bootstrap 95% confidence interval for a (y, s) -> float metric.

    Parameters
    ----------
    metric_fn : callable
        ``f(y_subset, s_subset) -> float``.
    y, s : numpy.ndarray
        Aligned labels and scores.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducible resampling.

    Returns
    -------
    tuple of (float, float)
        ``(lo, hi)`` 2.5/97.5 percentile bounds of the metric distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            out[i] = float(metric_fn(y[idx], s[idx]))
        except (ValueError, ZeroDivisionError):
            out[i] = np.nan
    lo = float(np.nanpercentile(out, 2.5))
    hi = float(np.nanpercentile(out, 97.5))
    return lo, hi


def evaluate_scores(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    seed: int = 42,
) -> MetricsResult:
    """Compute the full metric bundle (with bootstrap CIs) for a single model.

    Parameters
    ----------
    y : numpy.ndarray
        Test labels (0/1).
    scores : numpy.ndarray
        Test scores in [0, 1] aligned with ``y``.
    threshold : float
        Operating-point threshold for F1 / confusion (fixed from val).
    n_boot : int
        Bootstrap resamples for the AUROC/AUPRC/F1 95% CIs.
    seed : int
        Random seed.

    Returns
    -------
    MetricsResult
        Full evaluation bundle.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)
    auroc = float(roc_auc_score(y, scores))
    auprc = float(average_precision_score(y, scores))
    preds = (scores >= threshold).astype(int)
    f1 = float(f1_score(y, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    precision, recall, specificity, accuracy = _confusion_components(int(tn), int(fp), int(fn), int(tp))

    auroc_ci = _bootstrap_ci(roc_auc_score, y, scores, n_boot=n_boot, seed=seed)
    auprc_ci = _bootstrap_ci(average_precision_score, y, scores, n_boot=n_boot, seed=seed)

    def _f1_at_t(yy, ss) -> float:
        pp = (ss >= threshold).astype(int)
        return float(f1_score(yy, pp, zero_division=0))

    f1_ci = _bootstrap_ci(_f1_at_t, y, scores, n_boot=n_boot, seed=seed)

    return MetricsResult(
        n=int(len(y)),
        n_pos=int((y == 1).sum()),
        n_neg=int((y == 0).sum()),
        auroc=auroc,
        auprc=auprc,
        f1=f1,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        threshold=float(threshold),
        auroc_ci=auroc_ci,
        auprc_ci=auprc_ci,
        f1_ci=f1_ci,
        confusion=(int(tn), int(fp), int(fn), int(tp)),
    )


def per_gene_auroc(
    y: np.ndarray,
    scores: np.ndarray,
    genes: np.ndarray,
    min_per_class: int = 5,
) -> dict[str, float]:
    """Compute per-gene AUROC for genes with enough variants of both classes.

    Parameters
    ----------
    y, scores : numpy.ndarray
        Aligned labels and scores.
    genes : numpy.ndarray
        Gene symbols aligned with ``y``.
    min_per_class : int
        Minimum number of pos AND neg variants required to compute AUROC.

    Returns
    -------
    dict[str, float]
        Mapping gene -> AUROC. Genes failing the size requirement are
        omitted.
    """
    out: dict[str, float] = {}
    unique_genes = np.unique(genes)
    for g in unique_genes:
        mask = genes == g
        if mask.sum() < 2 * min_per_class:
            continue
        ys = y[mask]
        ss = scores[mask]
        if int((ys == 1).sum()) < min_per_class or int((ys == 0).sum()) < min_per_class:
            continue
        try:
            out[str(g)] = float(roc_auc_score(ys, ss))
        except ValueError:
            continue
    return out
