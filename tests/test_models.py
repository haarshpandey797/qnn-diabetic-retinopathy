"""Tests that need TensorFlow and/or Qiskit; they are skipped when those are missing."""

import json

import numpy as np
import pytest


def test_autoencoder_encodes_to_one_feature_per_qubit():
    pytest.importorskip("tensorflow")
    from qnn_dr.features import encode, fit_encoder, scale_to_angles

    images = np.random.default_rng(0).uniform(size=(16, 28, 28, 3)).astype("float32")
    encoder, _ = fit_encoder(images, latent_dim=8, epochs=1, verbose=0)
    feats = encode(encoder, images)
    assert feats.shape == (16, 8)
    train_a, test_a = scale_to_angles(feats, feats + 10)
    assert train_a.min() >= 0 and train_a.max() <= np.pi
    assert np.allclose(test_a, np.pi)


def test_qcnn_structure():
    pytest.importorskip("qiskit_machine_learning")
    from qnn_dr.qcnn import build_qcnn

    circuit, feature_map, ansatz, observables, readout = build_qcnn(8, 5)
    assert circuit.num_qubits == 8
    assert feature_map.num_parameters == 8
    # c1: 8 blocks x 3, p1: 3 blocks x 3, c2: 4 blocks x 3
    assert ansatz.num_parameters == 24 + 9 + 12
    assert readout == [3, 4, 5, 6, 7]
    assert len(observables) == 5


def _inputs(encoding, n, rng):
    if encoding == "amplitude":
        v = rng.normal(size=(n, 256))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return rng.uniform(0, np.pi, size=(n, 8))


@pytest.mark.parametrize("encoding,reuploads", [("angle", 1), ("angle", 3), ("amplitude", 1)])
def test_statevector_qnn_matches_estimator_qnn(encoding, reuploads):
    pytest.importorskip("qiskit_machine_learning")
    from qnn_dr.qcnn import build_qnn

    rng = np.random.default_rng(1)
    x = _inputs(encoding, 6, rng)
    reference, _ = build_qnn(encoding=encoding, reuploads=reuploads, simulator="estimator")
    fast, _ = build_qnn(encoding=encoding, reuploads=reuploads, simulator="statevector")
    assert fast.num_weights == reference.num_weights
    w = rng.uniform(-np.pi, np.pi, size=fast.num_weights)
    expected, got = reference.forward(x, w), fast.forward(x, w)
    assert got.shape == expected.shape == (6, 5)
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_reuploads_add_one_conv_layer_each():
    pytest.importorskip("qiskit_machine_learning")
    from qnn_dr.qcnn import build_qnn

    counts = [build_qnn(reuploads=r)[0].num_weights for r in (1, 2, 3)]
    assert counts == [45, 69, 93]


def test_cross_entropy_outputs_are_probabilities():
    pytest.importorskip("qiskit_machine_learning")
    from qnn_dr.qcnn import build_qnn

    qnn, _ = build_qnn(loss="cross_entropy")
    assert qnn.num_weights == 46  # 45 angles + log softmax scale
    rng = np.random.default_rng(2)
    p = qnn.forward(_inputs("angle", 4, rng), rng.uniform(-1, 1, size=46))
    assert np.all(p >= 0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "simulator,encoding,loss",
    [
        ("statevector", "angle", "squared_error"),
        ("estimator", "angle", "squared_error"),
        ("statevector", "amplitude", "cross_entropy"),
    ],
)
def test_qcnn_trains_and_predicts(simulator, encoding, loss):
    pytest.importorskip("qiskit_machine_learning")
    from qnn_dr.qcnn import build_classifier, predict

    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5), 2)
    x = _inputs(encoding, 10, rng)
    losses = []
    clf, _ = build_classifier(maxiter=5, callback=lambda w, v: losses.append(v),
                              simulator=simulator, encoding=encoding, loss=loss)
    clf.fit(x, y)
    pred = predict(clf, x)
    assert pred.shape == (10,)
    assert set(pred.tolist()) <= set(range(5))
    assert len(losses) > 0


@pytest.mark.parametrize(
    "features,size,extra",
    [
        ("autoencoder", "28", []),
        ("pretrained", "32", []),
        ("pretrained", "32", ["--encoding", "amplitude", "--loss", "cross_entropy"]),
        ("pretrained", "32", ["--reuploads", "2"]),
        ("pretrained", "32", ["--model", "qsvc"]),
    ],
)
def test_train_script_end_to_end(fake_idrid, tmp_path, features, size, extra):
    pytest.importorskip("tensorflow")
    pytest.importorskip("qiskit_machine_learning")
    import train

    out = tmp_path / "results"
    train.main([
        "--data-dir", str(fake_idrid), "--out", str(out), "--features", features,
        "--img-size", size, "--ae-epochs", "1", "--maxiter", "3", "--cache-dir", str(tmp_path / "cache"),
        *extra,
    ])
    summary = json.loads((out / "metrics.json").read_text())
    model = "qsvc" if "qsvc" in extra else "qcnn"
    assert {"majority_class", "logistic_regression", "mlp", "rbf_svm", model} <= set(summary["test_results"])
    assert (out / f"confusion_matrix_{model}.png").exists()
    if features == "pretrained":
        assert "logreg_all_2048_features" in summary["test_results"]


def test_cross_validation_uses_training_rows_only(fake_idrid, tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("qiskit_machine_learning")
    import cv

    out = tmp_path / "cv"
    agg = cv.main([
        "--data-dir", str(fake_idrid), "--out", str(out), "--features", "pretrained", "--img-size", "32",
        "--maxiter", "3", "--folds", "2", "--cache-dir", str(tmp_path / "cache"),
    ])
    assert agg is not None and "qcnn" in agg
    folds = [json.loads((out / f"fold_{k}.json").read_text()) for k in range(2)]
    assert sum(f["eval_size"] for f in folds) == 20  # the 20 training images, never the 10 test images


def test_quantum_kernel_svm_fits_and_predicts():
    pytest.importorskip("qiskit_machine_learning")
    from qnn_dr.qkernel import BandwidthQSVC

    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5), 4)
    x = rng.uniform(0, np.pi, size=(20, 8)) * 0.2 + y[:, None] * 0.5
    model = BandwidthQSVC(bandwidth=0.25, C=10.0).fit(x, y)
    pred = model.predict(x)
    assert pred.shape == (20,)
    assert (pred == y).mean() > 0.5  # separable toy data


def test_bandwidth_controls_kernel_concentration():
    pytest.importorskip("qiskit_machine_learning")
    from qnn_dr.qkernel import BandwidthQSVC

    x = np.random.default_rng(1).uniform(0, np.pi, size=(12, 8))
    off_diagonal = []
    for bw in (1.0, 0.25):
        m = BandwidthQSVC(feature_map="zz", bandwidth=bw)
        k = m.svc.quantum_kernel.evaluate(bw * x)
        off_diagonal.append(k[~np.eye(12, dtype=bool)].mean())
    assert off_diagonal[0] < off_diagonal[1]  # wider angles -> states closer to orthogonal
