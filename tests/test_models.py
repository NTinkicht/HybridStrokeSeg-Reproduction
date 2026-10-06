import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from hybridstrokeseg.models import SVMConfig, make_rbf_svm


def test_svm_scaler_modes_are_explicit() -> None:
    standard = make_rbf_svm(SVMConfig(scaler="standard"))
    minmax = make_rbf_svm(SVMConfig(scaler="minmax"))
    unscaled = make_rbf_svm(SVMConfig(scaler="none"))

    assert isinstance(standard.named_steps["scale"], StandardScaler)
    assert isinstance(minmax.named_steps["scale"], MinMaxScaler)
    assert unscaled.named_steps["scale"] == "passthrough"


def test_all_svm_scaler_modes_fit_small_problem() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, 0.3],
            [1.0, 1.0],
            [0.8, 0.9],
            [0.9, 0.7],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    for scaler in ("standard", "minmax", "none"):
        model = make_rbf_svm(SVMConfig(C=1.0, gamma="scale", scaler=scaler))
        predictions = model.fit(X, y).predict(X)
        assert predictions.shape == y.shape
