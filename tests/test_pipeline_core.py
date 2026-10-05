import numpy as np

from hybridstrokeseg.metrics import evaluate_binary_segmentation
from hybridstrokeseg.models import MLPConfig, SVMConfig, make_mlp, make_rbf_svm
from hybridstrokeseg.pipeline import prepare_slice, restore_slice_prediction, select_slice_index
from hybridstrokeseg.preprocessing import PreprocessConfig, preprocess_flair
from hybridstrokeseg.sampling import balanced_binary_sample
from hybridstrokeseg.splits import make_patient_split


def test_patient_split_is_deterministic_and_disjoint():
    ids = [f"case-{index:02d}" for index in range(28)]
    split_a = make_patient_split(ids, train_size=19, seed=7)
    split_b = make_patient_split(ids, train_size=19, seed=7)
    assert split_a == split_b
    assert len(split_a.train_ids) == 19
    assert len(split_a.test_ids) == 9
    assert set(split_a.train_ids).isdisjoint(split_a.test_ids)


def test_balanced_binary_sample_is_exactly_balanced():
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.array([1] * 6 + [0] * 14)
    sample = balanced_binary_sample(X, y, target_per_class=5, seed=3)
    assert sample.X.shape == (10, 3)
    assert sample.positive_count == 5
    assert sample.negative_count == 5
    assert int(sample.y.sum()) == 5


def test_max_lesion_slice_is_oracle_largest_area():
    flair = np.ones((4, 5, 5), dtype=float)
    lesion = np.zeros_like(flair)
    lesion[1, :2, :2] = 1
    lesion[3, :3, :3] = 1
    assert select_slice_index(flair, lesion, strategy="max_lesion") == 3


def test_prepare_and_restore_round_trip_without_closing():
    flair = np.array([[0, 2, 2], [0, 3, 4], [0, 0, 5]], dtype=float)
    lesion = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=int)
    prepared = prepare_slice(flair, lesion)
    restored = restore_slice_prediction(prepared.y, prepared.brain_mask, close_radius=0)
    assert prepared.X.shape[1] == 9
    assert np.array_equal(restored, (lesion > 0) & prepared.brain_mask)


def test_robust_uint8_preprocessing_has_expected_range():
    flair = np.arange(100, dtype=float).reshape(10, 10)
    processed, brain = preprocess_flair(
        flair,
        PreprocessConfig(mode="robust_uint8", median_size=1),
    )
    assert brain.any()
    assert float(processed.min()) >= 0.0
    assert float(processed.max()) <= 255.0


def test_metrics_use_continuous_scores_for_auc():
    gt = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.4, 0.8, 0.9])
    metrics = evaluate_binary_segmentation(gt, pred, scores=scores)
    assert np.isclose(metrics.dice, 0.8)
    assert np.isclose(metrics.auc, 1.0)


def test_models_fit_small_fixture():
    X = np.array(
        [
            [-2.0, -2.0],
            [-1.5, -1.0],
            [-1.0, -1.5],
            [1.0, 1.5],
            [1.5, 1.0],
            [2.0, 2.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    mlp = make_mlp(MLPConfig(hidden_layers=(4,), max_iter=200, random_state=1))
    svm = make_rbf_svm(SVMConfig(C=10.0, gamma="scale"))
    mlp.fit(X, y)
    svm.fit(X, y)
    assert (mlp.predict(X) == y).mean() >= 0.8
    assert (svm.predict(X) == y).mean() >= 0.8
