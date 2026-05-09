"""Probability calibration: reliability curves, Brier score, and ECE.

Expected calibration error (ECE) is the sample-weighted mean absolute gap
between bin-wise observed prevalence (accuracy) and mean predicted
probability (confidence), using quantile binning consistent with
``sklearn.calibration.calibration_curve(..., strategy='quantile')``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


@dataclass(frozen=True)
class CalibrationMetrics:
    """Reliability-curve coordinates plus scalar calibration scores.

    Attributes
    ----------
    prob_true : numpy.ndarray
        Fraction of positives in each bin (reliability diagram vertical axis).
    prob_pred : numpy.ndarray
        Mean predicted score in each bin (horizontal axis).
    brier : float
        Brier score (mean squared error between labels and probabilities).
    ece : float
        Expected calibration error.
    """

    prob_true: np.ndarray
    prob_pred: np.ndarray
    brier: float
    ece: float


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Compute ECE as the weighted average |accuracy − confidence| across bins.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary labels in ``{0, 1}``.
    y_prob : numpy.ndarray
        Predicted probabilities (values are clipped to ``[0, 1]``).
    n_bins : int
        Number of bins (default 10, matching the report appendix).
    strategy : str
        ``"quantile"`` (equal-mass bins) or ``"uniform"`` on ``[0, 1]``.

    Returns
    -------
    float
        ECE in ``[0, 1]``.
    """
    y_true = np.asarray(y_true).astype(int, copy=False)
    y_prob = np.clip(np.asarray(y_prob).astype(float, copy=False), 0.0, 1.0)
    n = int(len(y_true))
    if n == 0:
        return 0.0

    if strategy == "quantile":
        order = np.argsort(y_prob)
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n, dtype=np.int64)
        bin_id = np.minimum((ranks * n_bins) // n, n_bins - 1)
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_id = np.digitize(y_prob, edges[1:-1], right=False)
        bin_id = np.clip(bin_id, 0, n_bins - 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ece = 0.0
    for b in range(n_bins):
        mask = bin_id == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += abs(acc - conf) * (float(np.sum(mask)) / n)
    return float(ece)


def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> CalibrationMetrics:
    """Build reliability diagram coordinates plus Brier score and ECE.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary labels.
    y_prob : numpy.ndarray
        Predicted probabilities.
    n_bins : int
        Number of bins for :func:`sklearn.calibration.calibration_curve`.
    strategy : str
        ``"quantile"`` or ``"uniform"`` (passed through to sklearn).

    Returns
    -------
    CalibrationMetrics
        Bundled diagnostics for one model on one split.
    """
    y_true = np.asarray(y_true).astype(int, copy=False)
    y_prob = np.clip(np.asarray(y_prob).astype(float, copy=False), 0.0, 1.0)
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
    )
    brier = float(brier_score_loss(y_true, y_prob))
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    return CalibrationMetrics(
        prob_true=np.asarray(prob_true, dtype=float),
        prob_pred=np.asarray(prob_pred, dtype=float),
        brier=brier,
        ece=ece,
    )
