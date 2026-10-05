"""Classical models for the manuscript reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class MLPConfig:
    """Paper-reported MLP architecture with an explicit optimizer substitution."""

    hidden_layers: tuple[int, ...] = (100, 100, 100)
    activation: Literal["logistic", "tanh", "relu"] = "logistic"
    solver: Literal["lbfgs", "adam", "sgd"] = "lbfgs"
    max_iter: int = 500
    random_state: int = 2026


@dataclass(frozen=True)
class SVMConfig:
    """RBF-SVM defaults for parameters absent from the manuscript."""

    C: float = 1.0
    gamma: str | float = "scale"


def make_mlp(config: MLPConfig | None = None) -> Pipeline:
    """Build a scaled 3x100 sigmoid MLP.

    The manuscript reports scaled conjugate-gradient training. scikit-learn does
    not implement SCG, so the default LBFGS solver is a documented substitution.
    """
    cfg = config or MLPConfig()
    model = MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layers,
        activation=cfg.activation,
        solver=cfg.solver,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
    )
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def make_rbf_svm(config: SVMConfig | None = None) -> Pipeline:
    """Build a scaled RBF-SVM using libsvm's SMO-family optimization."""
    cfg = config or SVMConfig()
    model = SVC(kernel="rbf", C=cfg.C, gamma=cfg.gamma)
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def continuous_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return continuous positive-class scores for ROC/AUC."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64).reshape(-1)
    raise TypeError("Model provides neither decision_function nor predict_proba")
