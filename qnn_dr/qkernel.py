"""Quantum kernel classifier: a fidelity kernel from a Qiskit feature map, used by an SVM.

The kernel between two images is |<phi(x)|phi(y)>|^2, where |phi(x)> is the
8-qubit state the feature map prepares from the image's 8 features. Inputs are
multiplied by `bandwidth` first: at full scale (angles up to pi) the states of
different images are nearly orthogonal, the kernel matrix approaches the
identity, and the SVM cannot generalise. Shrinking the angles fixes that.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

FEATURE_MAPS = {"z": ZFeatureMap, "zz": ZZFeatureMap}


class BandwidthQSVC:
    """QSVC on bandwidth-scaled inputs, with balanced class weights."""

    def __init__(self, num_qubits: int = 8, feature_map: str = "z", bandwidth: float = 0.25, C: float = 10.0):
        self.bandwidth = bandwidth
        self.feature_map = FEATURE_MAPS[feature_map](num_qubits, reps=2)
        kernel = FidelityStatevectorKernel(feature_map=self.feature_map)
        self.svc = QSVC(quantum_kernel=kernel, C=C, class_weight="balanced")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BandwidthQSVC":
        self.svc.fit(self.bandwidth * np.asarray(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.svc.predict(self.bandwidth * np.asarray(x))).astype(int)
