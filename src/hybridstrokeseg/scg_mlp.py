"""Scaled-conjugate-gradient MLP for the historical paper reproduction.

The manuscript explicitly reports scaled conjugate-gradient backpropagation but
its original code is unavailable. This module implements Møller's SCG update and
a paper-shaped 9 -> 100 -> 100 -> 100 -> 2 sigmoid network in NumPy.

Important: the manuscript does not state the loss function, weight initializer,
SCG stopping criteria, or exact input scaling. Those remain explicit
reconstruction choices; the optimizer itself no longer needs to be substituted
with LBFGS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SCGConfig:
    """Configuration for Møller scaled conjugate gradient.

    ``sigma`` and ``lambda_initial`` use the long-standing MATLAB ``trainscg``
    defaults. The original manuscript does not report whether those defaults
    were changed.
    """

    max_iter: int = 500
    min_grad: float = 1e-6
    sigma: float = 5.0e-5
    lambda_initial: float = 5.0e-7
    goal: float = 0.0

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.min_grad <= 0:
            raise ValueError("min_grad must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.lambda_initial <= 0:
            raise ValueError("lambda_initial must be positive")


@dataclass(frozen=True)
class SCGResult:
    weights: np.ndarray
    loss_history: tuple[float, ...]
    gradient_norm_history: tuple[float, ...]
    iterations: int
    converged: bool
    stop_reason: str


def scaled_conjugate_gradient(
    initial_weights: np.ndarray,
    loss_and_gradient,
    *,
    config: SCGConfig | None = None,
) -> SCGResult:
    """Minimize a differentiable scalar objective using Møller's SCG method.

    ``loss_and_gradient(weights)`` must return ``(loss, gradient)`` where the
    gradient has the same shape as ``weights``.
    """
    cfg = config or SCGConfig()
    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    loss, gradient = loss_and_gradient(weights)
    loss = float(loss)
    gradient = np.asarray(gradient, dtype=np.float64)
    if gradient.shape != weights.shape:
        raise ValueError("loss_and_gradient returned a gradient with the wrong shape")
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Initial SCG loss/gradient is not finite")

    residual = -gradient
    direction = residual.copy()
    success = True
    lambda_value = float(cfg.lambda_initial)
    lambda_bar = 0.0
    n_parameters = max(1, weights.size)

    loss_history = [loss]
    grad_norm_history = [float(np.linalg.norm(gradient))]
    stop_reason = "max_iter"
    converged = False

    # Curvature quantities are intentionally retained after an unsuccessful
    # step, exactly as in the SCG recurrence where they are recomputed only
    # after a successful step.
    delta = 0.0
    kappa = 0.0
    mu = 0.0

    for iteration in range(1, cfg.max_iter + 1):
        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm <= cfg.min_grad:
            converged = True
            stop_reason = "min_grad"
            break
        if loss <= cfg.goal:
            converged = True
            stop_reason = "goal"
            break

        if success:
            kappa = float(np.dot(direction, direction))
            if kappa <= np.finfo(np.float64).eps:
                converged = True
                stop_reason = "zero_direction"
                break

            sigma_k = cfg.sigma / np.sqrt(kappa)
            _, gradient_sigma = loss_and_gradient(weights + sigma_k * direction)
            gradient_sigma = np.asarray(gradient_sigma, dtype=np.float64)
            second_order = (gradient_sigma - gradient) / sigma_k
            delta = float(np.dot(direction, second_order))

        # Scale the Hessian approximation.
        delta += (lambda_value - lambda_bar) * kappa

        if delta <= 0.0:
            lambda_bar = 2.0 * (lambda_value - delta / kappa)
            delta = -delta + lambda_value * kappa
            lambda_value = lambda_bar

        mu = float(np.dot(direction, residual))
        if mu <= 0.0:
            # Numerical safeguards/restarts are required when finite precision
            # destroys descent. This is equivalent to a steepest-descent restart.
            direction = residual.copy()
            kappa = float(np.dot(direction, direction))
            mu = kappa
            success = True
            lambda_bar = 0.0
            if kappa <= np.finfo(np.float64).eps:
                converged = True
                stop_reason = "zero_residual"
                break

            sigma_k = cfg.sigma / np.sqrt(kappa)
            _, gradient_sigma = loss_and_gradient(weights + sigma_k * direction)
            gradient_sigma = np.asarray(gradient_sigma, dtype=np.float64)
            second_order = (gradient_sigma - gradient) / sigma_k
            delta = float(np.dot(direction, second_order)) + lambda_value * kappa
            if delta <= 0.0:
                lambda_bar = 2.0 * (lambda_value - delta / kappa)
                delta = -delta + lambda_value * kappa
                lambda_value = lambda_bar

        alpha = mu / delta
        candidate = weights + alpha * direction
        candidate_loss, candidate_gradient = loss_and_gradient(candidate)
        candidate_loss = float(candidate_loss)
        candidate_gradient = np.asarray(candidate_gradient, dtype=np.float64)

        if not np.isfinite(candidate_loss) or not np.isfinite(candidate_gradient).all():
            comparison = -np.inf
        else:
            comparison = 2.0 * delta * (loss - candidate_loss) / (mu * mu)

        if comparison >= 0.0:
            weights = candidate
            loss = candidate_loss
            old_residual = residual
            gradient = candidate_gradient
            residual = -gradient
            lambda_bar = 0.0
            success = True

            if iteration % n_parameters == 0:
                direction = residual.copy()
            else:
                beta = (
                    float(np.dot(residual, residual))
                    - float(np.dot(residual, old_residual))
                ) / mu
                direction = residual + beta * direction

            if comparison >= 0.75:
                lambda_value *= 0.25
        else:
            lambda_bar = lambda_value
            success = False

        if comparison < 0.25:
            lambda_value += delta * (1.0 - comparison) / kappa

        # Avoid exact zero/overflow without imposing a meaningful user tuning
        # parameter. MATLAB's legacy implementation similarly maintains a
        # positive Hessian-regularization scale internally.
        lambda_value = float(np.clip(lambda_value, 1e-30, 1e30))

        loss_history.append(loss)
        grad_norm_history.append(float(np.linalg.norm(gradient)))
    else:
        iteration = cfg.max_iter

    return SCGResult(
        weights=weights,
        loss_history=tuple(loss_history),
        gradient_norm_history=tuple(grad_norm_history),
        iterations=int(iteration),
        converged=converged,
        stop_reason=stop_reason,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class PaperSCGMLPClassifier(ClassifierMixin, BaseEstimator):
    """Two-output sigmoid MLP trained with scaled conjugate gradient.

    The output encoding follows the manuscript: lesion -> ``[1, 0]`` and
    non-lesion -> ``[0, 1]``. ``predict_proba`` reorders the two activations to
    sklearn's conventional class order ``[non-lesion, lesion]`` and normalizes
    them to sum to one.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (100, 100, 100),
        loss: Literal["cross_entropy", "mse"] = "cross_entropy",
        max_iter: int = 500,
        min_grad: float = 1e-6,
        sigma: float = 5.0e-5,
        lambda_initial: float = 5.0e-7,
        random_state: int = 2026,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.max_iter = max_iter
        self.min_grad = min_grad
        self.sigma = sigma
        self.lambda_initial = lambda_initial
        self.random_state = random_state

    def _validate_hyperparameters(self) -> None:
        if not self.hidden_layers or any(int(size) < 1 for size in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive sizes")
        if self.loss not in {"cross_entropy", "mse"}:
            raise ValueError("loss must be 'cross_entropy' or 'mse'")

    @staticmethod
    def _layout(layer_sizes: tuple[int, ...]) -> list[tuple[slice, tuple[int, ...], slice]]:
        layout: list[tuple[slice, tuple[int, ...], slice]] = []
        cursor = 0
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
            weight_count = fan_in * fan_out
            weight_slice = slice(cursor, cursor + weight_count)
            cursor += weight_count
            bias_slice = slice(cursor, cursor + fan_out)
            cursor += fan_out
            layout.append((weight_slice, (fan_in, fan_out), bias_slice))
        return layout

    @staticmethod
    def _unpack(
        vector: np.ndarray,
        layout: list[tuple[slice, tuple[int, ...], slice]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        parameters: list[tuple[np.ndarray, np.ndarray]] = []
        for weight_slice, shape, bias_slice in layout:
            weights = vector[weight_slice].reshape(shape)
            biases = vector[bias_slice]
            parameters.append((weights, biases))
        return parameters

    def _initialize(
        self,
        layer_sizes: tuple[int, ...],
        layout: list[tuple[slice, tuple[int, ...], slice]],
    ) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        total = layout[-1][2].stop
        vector = np.zeros(total, dtype=np.float64)
        for weight_slice, (fan_in, fan_out), bias_slice in layout:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            vector[weight_slice] = rng.uniform(-limit, limit, size=fan_in * fan_out)
            vector[bias_slice] = 0.0
        return vector

    def _targets(self, y: np.ndarray) -> np.ndarray:
        targets = np.empty((y.size, 2), dtype=np.float64)
        targets[:, 0] = y  # lesion output, matching manuscript [1, 0]
        targets[:, 1] = 1 - y
        return targets

    def _forward(
        self,
        X: np.ndarray,
        vector: np.ndarray,
        layout: list[tuple[slice, tuple[int, ...], slice]],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        preactivations: list[np.ndarray] = []
        current = X
        for weights, biases in self._unpack(vector, layout):
            z = current @ weights + biases
            current = _sigmoid(z)
            preactivations.append(z)
            activations.append(current)
        return activations, preactivations

    def _objective(
        self,
        X: np.ndarray,
        targets: np.ndarray,
        vector: np.ndarray,
        layout: list[tuple[slice, tuple[int, ...], slice]],
    ) -> tuple[float, np.ndarray]:
        activations, _ = self._forward(X, vector, layout)
        output = activations[-1]
        n_samples = X.shape[0]

        if self.loss == "cross_entropy":
            eps = 1e-12
            clipped = np.clip(output, eps, 1.0 - eps)
            loss = -np.mean(
                np.sum(
                    targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped),
                    axis=1,
                )
            )
            delta = (output - targets) / n_samples
        else:
            error = output - targets
            loss = 0.5 * np.mean(np.sum(error * error, axis=1))
            delta = error * output * (1.0 - output) / n_samples

        gradients: list[tuple[np.ndarray, np.ndarray]] = []
        parameters = self._unpack(vector, layout)
        for layer_index in range(len(parameters) - 1, -1, -1):
            weights, _ = parameters[layer_index]
            previous_activation = activations[layer_index]
            grad_w = previous_activation.T @ delta
            grad_b = np.sum(delta, axis=0)
            gradients.append((grad_w, grad_b))

            if layer_index > 0:
                previous_output = activations[layer_index]
                delta = (delta @ weights.T) * previous_output * (1.0 - previous_output)

        gradients.reverse()
        gradient_vector = np.empty_like(vector)
        for (weight_slice, _, bias_slice), (grad_w, grad_b) in zip(
            layout,
            gradients,
            strict=True,
        ):
            gradient_vector[weight_slice] = grad_w.reshape(-1)
            gradient_vector[bias_slice] = grad_b
        return float(loss), gradient_vector

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._validate_hyperparameters()
        features = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y, dtype=np.int64).reshape(-1)
        if features.ndim != 2:
            raise ValueError("X must be a 2-D feature matrix")
        if labels.shape[0] != features.shape[0]:
            raise ValueError("X and y contain different numbers of samples")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("PaperSCGMLPClassifier supports binary labels 0/1 only")
        if not np.isfinite(features).all():
            raise ValueError("X contains NaN or infinite values")

        self.n_features_in_ = features.shape[1]
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.layer_sizes_ = (
            self.n_features_in_,
            *(int(size) for size in self.hidden_layers),
            2,
        )
        self.layout_ = self._layout(self.layer_sizes_)
        initial = self._initialize(self.layer_sizes_, self.layout_)
        targets = self._targets(labels)

        def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
            return self._objective(features, targets, vector, self.layout_)

        result = scaled_conjugate_gradient(
            initial,
            objective,
            config=SCGConfig(
                max_iter=self.max_iter,
                min_grad=self.min_grad,
                sigma=self.sigma,
                lambda_initial=self.lambda_initial,
            ),
        )
        self.coef_vector_ = result.weights
        self.loss_curve_ = list(result.loss_history)
        self.gradient_norm_curve_ = list(result.gradient_norm_history)
        self.n_iter_ = result.iterations
        self.converged_ = result.converged
        self.stop_reason_ = result.stop_reason
        return self

    def _raw_outputs(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_vector_"):
            raise RuntimeError("PaperSCGMLPClassifier has not been fitted")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("X has incompatible shape")
        activations, _ = self._forward(features, self.coef_vector_, self.layout_)
        return activations[-1]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        output = self._raw_outputs(X)
        return output[:, 0] - output[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        output = self._raw_outputs(X)
        conventional = np.column_stack((output[:, 1], output[:, 0]))
        denominator = conventional.sum(axis=1, keepdims=True)
        denominator = np.maximum(denominator, np.finfo(np.float64).eps)
        return conventional / denominator

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(np.int64)


def make_scg_mlp(
    *,
    hidden_layers: tuple[int, ...] = (100, 100, 100),
    loss: Literal["cross_entropy", "mse"] = "cross_entropy",
    max_iter: int = 500,
    min_grad: float = 1e-6,
    sigma: float = 5.0e-5,
    lambda_initial: float = 5.0e-7,
    random_state: int = 2026,
) -> Pipeline:
    """Build the paper-shaped SCG MLP behind an explicit StandardScaler."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                PaperSCGMLPClassifier(
                    hidden_layers=hidden_layers,
                    loss=loss,
                    max_iter=max_iter,
                    min_grad=min_grad,
                    sigma=sigma,
                    lambda_initial=lambda_initial,
                    random_state=random_state,
                ),
            ),
        ]
    )
