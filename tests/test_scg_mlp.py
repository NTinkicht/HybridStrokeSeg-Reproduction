import numpy as np

from hybridstrokeseg.scg_mlp import (
    PaperSCGMLPClassifier,
    SCGConfig,
    scaled_conjugate_gradient,
)


def test_scg_converges_on_positive_quadratic():
    target = np.array([1.5, -2.0, 0.25], dtype=float)

    def objective(weights):
        error = weights - target
        return 0.5 * float(np.dot(error, error)), error

    result = scaled_conjugate_gradient(
        np.array([8.0, 5.0, -3.0]),
        objective,
        config=SCGConfig(max_iter=100, min_grad=1e-10),
    )

    assert result.loss_history[-1] < result.loss_history[0]
    assert np.allclose(result.weights, target, atol=1e-5)
    assert result.gradient_norm_history[-1] < 1e-5


def test_scg_mlp_gradient_matches_finite_difference():
    X = np.array(
        [
            [-1.0, -0.5],
            [-0.2, 0.3],
            [0.4, 0.8],
            [1.0, 0.5],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1], dtype=int)
    model = PaperSCGMLPClassifier(hidden_layers=(3,), random_state=3)
    model._validate_hyperparameters()
    layer_sizes = (2, 3, 2)
    layout = model._layout(layer_sizes)
    vector = model._initialize(layer_sizes, layout)
    targets = model._targets(y)
    _, analytic = model._objective(X, targets, vector, layout)

    epsilon = 1e-6
    indices = np.linspace(0, vector.size - 1, num=min(8, vector.size), dtype=int)
    for index in indices:
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        loss_plus, _ = model._objective(X, targets, plus, layout)
        loss_minus, _ = model._objective(X, targets, minus, layout)
        numeric = (loss_plus - loss_minus) / (2.0 * epsilon)
        assert np.isclose(analytic[index], numeric, rtol=2e-4, atol=2e-5)


def test_scg_mlp_learns_small_separable_fixture():
    X = np.array(
        [
            [-2.0, -1.5],
            [-1.5, -2.0],
            [-1.0, -1.0],
            [-0.8, -1.2],
            [0.8, 1.2],
            [1.0, 1.0],
            [1.5, 2.0],
            [2.0, 1.5],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    model = PaperSCGMLPClassifier(
        hidden_layers=(5,),
        max_iter=250,
        min_grad=1e-7,
        random_state=7,
    )
    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    assert (predictions == y).mean() >= 0.875
    assert probabilities.shape == (len(y), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.loss_curve_[-1] < model.loss_curve_[0]
