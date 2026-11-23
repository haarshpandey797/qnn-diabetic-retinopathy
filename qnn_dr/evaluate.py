"""Metrics, classical baselines and plots."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier

from .data import GRADE_NAMES, NUM_CLASSES


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """Accuracy alone is misleading on IDRiD (predicting grade 2 for every image
    already scores ~33%), so balanced accuracy, macro-F1 and quadratic weighted
    kappa (the usual DR-grading metric) are reported too."""
    labels = list(range(NUM_CLASSES))
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "quadratic_kappa": None if math.isnan(kappa) else float(kappa),
    }


def baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_train_original: np.ndarray,
    seed: int = 42,
) -> dict[str, dict]:
    """Classical models trained on exactly the same features as the QCNN.

    `y_train_original` is the label vector before oversampling, used for the
    majority-class baseline.
    """
    results = {}

    majority = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(y_train_original), 1)), y_train_original
    )
    results["majority_class"] = metrics(y_test, majority.predict(np.zeros((len(y_test), 1))))

    models = {
        "logistic_regression": LogisticRegression(max_iter=5000),
        "mlp": MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=seed),
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        results[name] = metrics(y_test, model.predict(x_test))
    return results


def save_confusion_matrix(y_true, y_pred, path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES), GRADE_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(NUM_CLASSES), GRADE_NAMES)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_objective_curve(values: list[float], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(values) + 1), values)
    ax.set_xlabel("COBYLA iteration")
    ax.set_ylabel("Training loss")
    ax.set_title("QCNN training objective")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_table(rows: dict[str, dict]) -> str:
    width = max(22, *(len(name) + 2 for name in rows))
    header = f"{'model':<{width}}{'acc':>8}{'bal_acc':>9}{'macro_f1':>10}{'qwk':>8}"
    lines = [header, "-" * len(header)]
    for name, m in rows.items():
        qwk = "n/a" if m["quadratic_kappa"] is None else f"{m['quadratic_kappa']:.3f}"
        lines.append(
            f"{name:<{width}}{m['accuracy']:>8.3f}{m['balanced_accuracy']:>9.3f}{m['macro_f1']:>10.3f}{qwk:>8}"
        )
    return "\n".join(lines)
