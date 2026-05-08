"""Logistic-regression and random-forest baselines on handcrafted features.

Both models use ``class_weight='balanced'`` to address the class imbalance
the proposal anticipates ("class weighting or balanced sampling during
training"). Hyperparameters are tuned on the validation split using a small
grid; the model with the highest validation AUPRC is retained per family.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from imc.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass
class TrainResult:
    """Container for a fitted classifier plus its tuning metadata.

    Attributes
    ----------
    model : sklearn estimator
        Best fitted estimator on the training split.
    name : str
        Short model name (``"lr"`` or ``"rf"``).
    best_params : dict
        Hyperparameter values that achieved the best validation AUPRC.
    val_auprc : float
        Validation AUPRC of ``best_params``.
    val_auroc : float
        Validation AUROC of ``best_params``.
    train_seconds : float
        Wall-clock training time of the best model (single fit, not whole grid).
    n_train : int
        Number of training rows used.
    """

    model: object
    name: str
    best_params: dict[str, object]
    val_auprc: float
    val_auroc: float
    train_seconds: float
    n_train: int


def _grid(param_grid: dict[str, list]) -> Iterable[dict[str, object]]:
    """Yield every combination from a sklearn-style param_grid dict.

    Parameters
    ----------
    param_grid : dict[str, list]
        Mapping from parameter name to candidate values.

    Yields
    ------
    dict[str, object]
        Each combination of values, one per yield.
    """
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    if not keys:
        yield {}
        return
    idx = [0] * len(keys)
    while True:
        yield {k: values[i][idx[i]] for i, k in enumerate(keys)}
        for j in range(len(idx) - 1, -1, -1):
            if idx[j] + 1 < len(values[j]):
                idx[j] += 1
                break
            idx[j] = 0
            if j == 0:
                return


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    C_grid: list[float],
    max_iter: int = 5000,
    seed: int = 42,
) -> TrainResult:
    """Train a logistic-regression baseline with a small ``C`` grid on val AUPRC.

    Parameters
    ----------
    X_train, y_train : numpy.ndarray
        Training features and labels.
    X_val, y_val : numpy.ndarray
        Validation features and labels.
    C_grid : list[float]
        Candidate inverse-regularization strengths.
    max_iter : int
        Maximum solver iterations.
    seed : int
        Random seed.

    Returns
    -------
    TrainResult
        Best fitted model and its validation metrics.
    """
    best: TrainResult | None = None
    for params in _grid({"C": C_grid}):
        t0 = time.time()
        clf = LogisticRegression(
            penalty="l2",
            C=float(params["C"]),
            class_weight="balanced",
            solver="liblinear",
            max_iter=max_iter,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0
        scores = clf.predict_proba(X_val)[:, 1]
        auprc = float(average_precision_score(y_val, scores))
        auroc = float(roc_auc_score(y_val, scores))
        LOG.info("LR C=%g | val AUPRC=%.4f AUROC=%.4f (%.1fs)", params["C"], auprc, auroc, elapsed)
        if best is None or auprc > best.val_auprc:
            best = TrainResult(
                model=clf,
                name="lr",
                best_params=params,
                val_auprc=auprc,
                val_auroc=auroc,
                train_seconds=elapsed,
                n_train=int(len(y_train)),
            )
    assert best is not None
    return best


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_estimators: int,
    max_depth_grid: list[object],
    min_samples_leaf_grid: list[int],
    n_jobs: int = -1,
    seed: int = 42,
) -> TrainResult:
    """Train a random-forest baseline with a small grid on val AUPRC.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : numpy.ndarray
        Train and validation features and labels.
    n_estimators : int
        Number of trees.
    max_depth_grid : list of (int or None)
        Candidate ``max_depth`` values (``None`` for unbounded).
    min_samples_leaf_grid : list[int]
        Candidate ``min_samples_leaf`` values.
    n_jobs : int
        Joblib parallelism (``-1`` = all cores).
    seed : int
        Random seed.

    Returns
    -------
    TrainResult
        Best fitted model and its validation metrics.
    """
    best: TrainResult | None = None
    for params in _grid({"max_depth": max_depth_grid, "min_samples_leaf": min_samples_leaf_grid}):
        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0
        scores = clf.predict_proba(X_val)[:, 1]
        auprc = float(average_precision_score(y_val, scores))
        auroc = float(roc_auc_score(y_val, scores))
        LOG.info(
            "RF max_depth=%s min_samples_leaf=%d | val AUPRC=%.4f AUROC=%.4f (%.1fs)",
            params["max_depth"], params["min_samples_leaf"], auprc, auroc, elapsed,
        )
        if best is None or auprc > best.val_auprc:
            best = TrainResult(
                model=clf,
                name="rf",
                best_params=params,
                val_auprc=auprc,
                val_auroc=auroc,
                train_seconds=elapsed,
                n_train=int(len(y_train)),
            )
    assert best is not None
    return best
