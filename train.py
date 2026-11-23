"""Train and evaluate the hybrid classical feature extractor + QCNN classifier on IDRiD.

Examples:
    python train.py --data-dir data/IDRiD --features autoencoder
    python train.py --data-dir data/IDRiD --features pretrained --encoding amplitude --loss cross_entropy
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from qnn_dr.data import LABEL_COLUMN, NUM_CLASSES, load_split, oversample, read_labels
from qnn_dr.evaluate import (
    baselines,
    format_table,
    metrics,
    save_confusion_matrix,
    save_objective_curve,
)

DEFAULT_IMG_SIZE = {"autoencoder": 28, "pretrained": 224}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/IDRiD", help="folder with Excel/, Training/ and Testing/")
    p.add_argument("--out", default=None, help="output folder (default: results/<features>)")
    p.add_argument("--features", choices=["autoencoder", "pretrained"], default="autoencoder",
                   help="autoencoder: conv autoencoder trained on IDRiD; pretrained: ImageNet ResNet50")
    p.add_argument("--model", choices=["qcnn", "qsvc"], default="qcnn",
                   help="qcnn: trainable quantum circuit; qsvc: quantum fidelity kernel + SVM")
    p.add_argument("--kernel-map", choices=["z", "zz"], default="z", help="qsvc: feature map of the kernel")
    p.add_argument("--bandwidth", type=float, default=0.25, help="qsvc: inputs are multiplied by this")
    p.add_argument("--svm-c", type=float, default=10.0, help="qsvc: SVM regularisation C")
    p.add_argument("--encoding", choices=["angle", "amplitude"], default="angle",
                   help="angle: one feature per qubit; amplitude: 2^qubits features in the state amplitudes")
    p.add_argument("--reuploads", type=int, default=1, help="how many times angle-encoded features enter the circuit")
    p.add_argument("--loss", choices=["squared_error", "cross_entropy"], default="squared_error",
                   help="cross_entropy adds a softmax with a trainable scale over the readout")
    p.add_argument("--img-size", type=int, default=None, help="default 28 (autoencoder) or 224 (pretrained)")
    p.add_argument("--qubits", type=int, default=8, help="number of qubits")
    p.add_argument("--ae-epochs", type=int, default=30, help="autoencoder training epochs")
    p.add_argument("--maxiter", type=int, default=300, help="COBYLA iterations for the QCNN")
    p.add_argument("--simulator", choices=["statevector", "estimator"], default="statevector",
                   help="fast exact StatevectorQNN or qiskit-machine-learning EstimatorQNN (slow)")
    p.add_argument("--no-balance", action="store_true", help="do not oversample minority grades")
    p.add_argument("--cache-dir", default=None, help="cache loaded images and ResNet50 features here")
    p.add_argument("--seed", type=int, default=42)
    return p


def finish_args(args):
    args.img_size = args.img_size or DEFAULT_IMG_SIZE[args.features]
    args.out = args.out or f"results/{args.features}"
    if args.encoding == "amplitude" and args.features != "pretrained":
        raise SystemExit("--encoding amplitude needs --features pretrained (the autoencoder gives one value per qubit)")
    if args.model == "qsvc" and args.encoding != "angle":
        raise SystemExit("--model qsvc uses angle features (one per qubit)")
    return args


def _cached(cache_dir: Path | None, name: str, compute):
    """Return compute(), reusing a saved .npz copy when cache_dir is set."""
    if cache_dir is None:
        return compute()
    path = cache_dir / f"{name}.npz"
    if path.exists():
        with np.load(path) as f:
            return f["arr"]
    arr = compute()
    np.savez_compressed(path, arr=arr)
    return arr


def extract_features(args, cache_dir):
    """Return (train_features, test_features, y_train, y_test).

    Autoencoder features have one column per qubit; pretrained features are the
    2048 ResNet50 values per image.
    """
    from qnn_dr.features import encode, fit_encoder, pretrained_features

    data_dir = Path(args.data_dir)
    splits = {
        "train": (data_dir / "Excel" / "Training_Labels.csv", data_dir / "Training"),
        "test": (data_dir / "Excel" / "Testing_Labels.csv", data_dir / "Testing"),
    }
    labels, images = {}, {}
    for name, (csv, folder) in splits.items():
        labels[name] = read_labels(csv)[LABEL_COLUMN].to_numpy()
        images[name] = _cached(
            cache_dir,
            f"images_{name}_{args.img_size}",
            lambda csv=csv, folder=folder: load_split(csv, folder, args.img_size).images,
        )
        print(f"  {name}: {images[name].shape}, grade counts "
              f"{np.bincount(labels[name], minlength=NUM_CLASSES).tolist()}")

    if args.features == "autoencoder":
        print("Training the convolutional autoencoder ...")
        encoder, _ = fit_encoder(images["train"], latent_dim=args.qubits, epochs=args.ae_epochs, seed=args.seed)
        feats = {name: encode(encoder, images[name]) for name in ("train", "test")}
    else:
        print("Extracting ImageNet ResNet50 features ...")
        feats = {
            name: _cached(cache_dir, f"resnet50_{name}_{args.img_size}",
                          lambda name=name: pretrained_features(images[name]))
            for name in ("train", "test")
        }
    return feats["train"], feats["test"], labels["train"], labels["test"]


def prepare_inputs(train_features, test_features, args):
    """Turn extracted features into QNN inputs. Everything is fitted on the training rows only."""
    from qnn_dr.features import reduce_to_qubits, scale_to_angles

    if args.encoding == "amplitude":
        dim = 2**args.qubits
        k = min(dim, *train_features.shape)  # PCA cannot return more components than samples
        a, b = reduce_to_qubits(train_features, test_features, k, seed=args.seed)
        a, b = (np.pad(v, ((0, 0), (0, dim - k))) for v in (a, b))
        normalise = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)  # noqa: E731
        return normalise(a), normalise(b)
    if train_features.shape[1] != args.qubits:
        train_features, test_features = reduce_to_qubits(train_features, test_features, args.qubits, seed=args.seed)
    return scale_to_angles(train_features, test_features)


def fit_and_evaluate(x_train, y_train_orig, x_test, y_test, args, full_train=None, full_test=None, verbose=True):
    """Fit the classical baselines and the quantum model on the training rows and
    score them on the test rows.

    Returns (summary, model, circuit, y_pred, objective).
    """
    from sklearn.svm import SVC

    index = np.arange(len(y_train_orig))
    y_bal = y_train_orig
    if not args.no_balance:
        index, y_bal = oversample(index, y_train_orig, seed=args.seed)
    x_bal = x_train[index]

    # Classical models on exactly the same inputs as the quantum model.
    results = baselines(x_bal, y_bal, x_test, y_test, y_train_orig, seed=args.seed)
    # RBF-SVM is the classical counterpart of the quantum kernel model; its
    # hyperparameters were chosen by cross-validation on the training set (cv.py).
    rbf = SVC(kernel="rbf", gamma=0.3, C=1.0, class_weight="balanced").fit(x_train, y_train_orig)
    results["rbf_svm"] = metrics(y_test, rbf.predict(x_test))
    if full_train is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        full_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        full_model.fit(full_train[index], y_bal)
        results[f"logreg_all_{full_train.shape[1]}_features"] = metrics(y_test, full_model.predict(full_test))

    summary = {"config": dict(vars(args)), "eval_size": int(len(y_test))}

    if args.model == "qsvc":
        from qnn_dr.qkernel import BandwidthQSVC

        model = BandwidthQSVC(args.qubits, args.kernel_map, args.bandwidth, args.svm_c)
        start = time.time()
        model.fit(x_train, y_train_orig)  # balanced class weights instead of oversampling
        summary["qsvc_fit_seconds"] = round(time.time() - start, 1)
        y_pred = model.predict(x_test)
        results["qsvc"] = metrics(y_test, y_pred)
        summary["qsvc_train_set_metrics"] = metrics(y_train_orig, model.predict(x_train))
        summary["test_results"] = results
        return summary, model, model.feature_map, y_pred, []

    from qnn_dr.qcnn import build_classifier, predict

    objective: list[float] = []

    def callback(_weights, value):
        objective.append(float(value))
        if verbose and len(objective) % 50 == 0:
            print(f"  iteration {len(objective):4d}  loss {value:.4f}", flush=True)

    classifier, circuit = build_classifier(
        num_qubits=args.qubits,
        num_classes=NUM_CLASSES,
        maxiter=args.maxiter,
        seed=args.seed,
        callback=callback,
        simulator=args.simulator,
        encoding=args.encoding,
        reuploads=args.reuploads,
        loss=args.loss,
    )
    start = time.time()
    classifier.fit(x_bal, y_bal)
    summary["qcnn_train_seconds"] = round(time.time() - start, 1)

    y_pred = predict(classifier, x_test)
    results["qcnn"] = metrics(y_test, y_pred)
    summary.update({
        "train_size_after_balancing": int(len(y_bal)),
        "qcnn_weights": int(classifier.neural_network.num_weights),
        "qcnn_iterations": len(objective),
        "qcnn_train_set_metrics": metrics(y_bal, predict(classifier, x_bal)),
        "test_results": results,
    })
    return summary, classifier, circuit, y_pred, objective


def main(argv=None) -> dict:
    args = finish_args(build_parser().parse_args(argv))
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading IDRiD from {args.data_dir} at {args.img_size}x{args.img_size} ...")
    train_f, test_f, y_train, y_test = extract_features(args, cache_dir)
    x_train, x_test = prepare_inputs(train_f, test_f, args)
    full = (train_f, test_f) if args.features == "pretrained" else (None, None)

    if args.model == "qsvc":
        print(f"Training baselines and the quantum kernel SVM ({args.kernel_map} feature map, "
              f"bandwidth={args.bandwidth}, C={args.svm_c}) ...")
    else:
        print(f"Training baselines and the QCNN ({args.encoding} encoding, reuploads={args.reuploads}, "
              f"loss={args.loss}, COBYLA maxiter={args.maxiter}, {args.simulator}) ...")
    summary, _, circuit, y_pred, objective = fit_and_evaluate(x_train, y_train, x_test, y_test, args, *full)

    save_confusion_matrix(y_test, y_pred, out / f"confusion_matrix_{args.model}.png",
                          f"{args.model.upper()} on IDRiD test set ({args.features} features)")
    if objective:
        save_objective_curve(objective, out / "qcnn_objective.png")
    try:
        circuit.draw("mpl", fold=-1).savefig(out / f"{args.model}_circuit.png", dpi=120, bbox_inches="tight")
    except Exception as exc:  # drawing needs pylatexenc; it is optional
        print(f"  (skipped circuit drawing: {exc})")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("\nTest-set results")
    print(format_table(summary["test_results"]))
    print(f"\nSaved metrics and plots to {out}/")
    return summary


if __name__ == "__main__":
    main()
