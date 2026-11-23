"""Quantum stage: a quantum convolutional neural network (QCNN) built with Qiskit.

QCNN head for the default 8 qubits and 5 classes:

    conv  c1 on qubits 0-7              two-qubit blocks on neighbouring pairs
    pool  p1: 0->3, 1->4, 2->5          qubits 0-2 are dropped after this
    conv  c2 on qubits 3-7
    readout <Z> on qubits 3-7           one value per DR grade

The image features enter the circuit in one of two ways:

    encoding="angle"      ZFeatureMap: one feature per qubit (8 features).
                          With reuploads=L, the features are encoded L times,
                          with a trainable convolution layer between encodings.
    encoding="amplitude"  the 256 amplitudes of the 8-qubit state hold a
                          normalised 256-value feature vector.

The predicted grade is the readout qubit with the largest value. With
loss="cross_entropy" the five values go through a softmax with a trainable
scale, so the outputs are class probabilities.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import EstimatorQNN, NeuralNetwork
from scipy.sparse import diags

try:  # qiskit-machine-learning >= 0.8
    from qiskit_machine_learning.optimizers import COBYLA
    from qiskit_machine_learning.utils import algorithm_globals
except ImportError:  # older releases depended on qiskit-algorithms
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals

ENCODE = "encode"  # marks where the input features enter the circuit


def conv_circuit(params) -> QuantumCircuit:
    """Two-qubit convolution block with 3 trainable angles."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def pool_circuit(params) -> QuantumCircuit:
    """Two-qubit pooling block: moves information from qubit 0 into qubit 1."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target


def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Apply conv_circuit to (0,1), (2,3), ... and then to (1,2), (3,4), ..."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    params = ParameterVector(prefix, length=num_qubits * 3)
    i = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[i : i + 3]), [q1, q2])
        qc.barrier()
        i += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        if q1 == q2:
            continue
        qc = qc.compose(conv_circuit(params[i : i + 3]), [q1, q2])
        qc.barrier()
        i += 3

    wrapped = QuantumCircuit(num_qubits)
    wrapped.append(qc.to_instruction(), qubits)
    return wrapped


def pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    """Pool each source qubit into its sink; the sources are not used afterwards."""
    if len(sources) != len(sinks):
        raise ValueError("pool_layer needs one sink per source")
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(prefix, length=len(sources) * 3)
    for k, (source, sink) in enumerate(zip(sources, sinks)):
        qc = qc.compose(pool_circuit(params[3 * k : 3 * k + 3]), [source, sink])
        qc.barrier()

    wrapped = QuantumCircuit(num_qubits)
    wrapped.append(qc.to_instruction(), range(num_qubits))
    return wrapped


def qcnn_head(num_qubits: int = 8, num_classes: int = 5):
    """Return (ansatz, observables, readout_qubits) for conv -> pool -> conv."""
    n_drop = num_qubits - num_classes
    if not 0 < n_drop <= num_classes:
        raise ValueError("need num_classes < num_qubits <= 2 * num_classes")
    sources = list(range(n_drop))
    sinks = list(range(n_drop, 2 * n_drop))
    readout = list(range(n_drop, num_qubits))

    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(pool_layer(sources, sinks, "p1"), range(2 * n_drop), inplace=True)
    ansatz.compose(conv_layer(num_classes, "c2"), readout, inplace=True)

    observables = [
        SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits=num_qubits) for q in readout
    ]
    return ansatz, observables, readout


def build_qcnn(num_qubits: int = 8, num_classes: int = 5):
    """Return (circuit, feature_map, ansatz, observables, readout_qubits) for the
    basic model: ZFeatureMap followed by the QCNN head."""
    ansatz, observables, readout = qcnn_head(num_qubits, num_classes)
    feature_map = ZFeatureMap(num_qubits)
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)
    return circuit, feature_map, ansatz, observables, readout


def build_model(num_qubits: int = 8, num_classes: int = 5, encoding: str = "angle", reuploads: int = 1):
    """Describe a model variant.

    Returns (blocks, circuit, feature_map, input_params, weight_params, observables).
    `blocks` lists the circuit in order: ENCODE where the features enter, and the
    trainable sub-circuits in between. `circuit` is the equivalent full Qiskit circuit.
    """
    head, observables, _ = qcnn_head(num_qubits, num_classes)
    if encoding == "amplitude":
        if reuploads != 1:
            raise ValueError("amplitude encoding prepares the state once; use reuploads=1")
        feature_map = RawFeatureVector(2**num_qubits)
        blocks = [ENCODE, head]
    elif encoding == "angle":
        if reuploads < 1:
            raise ValueError("reuploads must be >= 1")
        feature_map = ZFeatureMap(num_qubits)
        blocks = []
        for i in range(reuploads - 1):
            blocks += [ENCODE, conv_layer(num_qubits, f"u{i}")]
        blocks += [ENCODE, head]
    else:
        raise ValueError(f"unknown encoding {encoding!r}")

    circuit = QuantumCircuit(num_qubits)
    for block in blocks:
        circuit.compose(feature_map if block is ENCODE else block, range(num_qubits), inplace=True)
    input_params = list(feature_map.parameters)
    inputs = set(input_params)
    weight_params = [p for p in circuit.parameters if p not in inputs]
    return blocks, circuit, feature_map, input_params, weight_params, observables


def _zfeature_qubit_unitaries(x: np.ndarray) -> np.ndarray:
    """Single-qubit unitaries of ZFeatureMap(reps=2) per sample and qubit, shape (N, n, 2, 2).

    On each qubit ZFeatureMap applies H, P(2x), H, P(2x), i.e. (P(2x) H)(P(2x) H).
    """
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    p = np.zeros(x.shape + (2, 2), dtype=complex)
    p[..., 0, 0] = 1.0
    p[..., 1, 1] = np.exp(2j * x)
    ph = p @ h
    return ph @ ph


class StatevectorQNN(NeuralNetwork):
    """Exact, batched simulation of a model from `build_model`, returning the
    expectation value of each diagonal (Z-type) observable, or a softmax over
    them when `softmax=True` (the last weight is then the log of the scale).

    It computes the same numbers as EstimatorQNN with an exact estimator, but
    much faster for training:
    - everything that depends only on the inputs (the encoded state, and the
      single-qubit encoding unitaries for re-uploads) is computed once and
      cached, because the training inputs do not change between iterations;
    - each trainable block is turned into one 2^n x 2^n unitary with Qiskit's
      Operator per iteration and applied to all samples in one matrix product.
    It provides no gradients, so it is meant for gradient-free optimisers such
    as COBYLA.
    """

    def __init__(self, num_qubits, blocks, observables, weight_params, encoding="angle", softmax=False):
        if blocks[0] is not ENCODE:
            raise ValueError("the first block must encode the inputs")
        diagonals = []
        for obs in observables:
            matrix = SparsePauliOp(obs).to_matrix(sparse=True)
            if (matrix - diags(matrix.diagonal())).count_nonzero():
                raise ValueError("StatevectorQNN only supports diagonal (Z-type) observables")
            diagonals.append(np.real(matrix.diagonal()))
        self._n = num_qubits
        self._blocks = blocks
        self._encoding = encoding
        self._weight_params = list(weight_params)
        self._softmax = softmax
        self._diagonals = np.stack(diagonals, axis=1)  # (2**n, num_observables)
        self._cache_key = None
        self._cache = None
        super().__init__(
            num_inputs=2**num_qubits if encoding == "amplitude" else num_qubits,
            num_weights=len(self._weight_params) + int(softmax),
            sparse=False,
            output_shape=len(observables),
        )

    def _apply_qubit_unitaries(self, psi: np.ndarray, unitaries: np.ndarray) -> np.ndarray:
        n, batch = self._n, psi.shape[0]
        psi = psi.reshape((batch,) + (2,) * n)
        for q in range(n):
            axis = n - q  # qubit q is bit q of the basis index (little-endian)
            psi = np.moveaxis(psi, axis, -1)
            psi = np.einsum("bij,b...j->b...i", unitaries[:, q], psi)
            psi = np.moveaxis(psi, -1, axis)
        return psi.reshape(batch, 2**n)

    def _encoded(self, x: np.ndarray):
        key = (x.shape, hash(x.tobytes()))
        if key != self._cache_key:
            if self._encoding == "amplitude":
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                start, unitaries = (x / np.where(norms > 0, norms, 1.0)).astype(complex), None
            else:
                unitaries = _zfeature_qubit_unitaries(x)
                zero = np.zeros((len(x), 2**self._n), dtype=complex)
                zero[:, 0] = 1.0
                start = self._apply_qubit_unitaries(zero, unitaries)
            self._cache_key, self._cache = key, (start, unitaries)
        return self._cache

    def _forward(self, input_data, weights):
        psi, unitaries = self._encoded(np.asarray(input_data, dtype=float))
        weights = np.asarray(weights, dtype=float)
        values = dict(zip(self._weight_params, weights))
        for block in self._blocks[1:]:
            if block is ENCODE:
                psi = self._apply_qubit_unitaries(psi, unitaries)
            else:
                bound = block.assign_parameters({p: values[p] for p in block.parameters})
                psi = psi @ Operator(bound).data.T
        out = (np.abs(psi) ** 2) @ self._diagonals
        if not self._softmax:
            return out
        logits = np.exp(weights[-1]) * out
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)

    def _backward(self, input_data, weights):
        return None, None


def build_qnn(
    num_qubits: int = 8,
    num_classes: int = 5,
    encoding: str = "angle",
    reuploads: int = 1,
    loss: str = "squared_error",
    simulator: str = "statevector",
):
    """Return (qnn, circuit) for a model variant."""
    blocks, circuit, _, input_params, weight_params, observables = build_model(
        num_qubits, num_classes, encoding, reuploads
    )
    if simulator == "statevector":
        qnn = StatevectorQNN(
            num_qubits, blocks, observables, weight_params, encoding, softmax=(loss == "cross_entropy")
        )
    elif simulator == "estimator":
        if loss != "squared_error":
            raise ValueError("loss='cross_entropy' needs simulator='statevector'")
        qnn = EstimatorQNN(
            circuit=circuit.decompose() if encoding == "angle" else circuit,
            estimator=Estimator(),  # exact expectation values, no shot noise
            observables=observables,
            input_params=input_params,
            weight_params=weight_params,
        )
    else:
        raise ValueError(f"unknown simulator {simulator!r}")
    return qnn, circuit


def build_classifier(
    num_qubits: int = 8,
    num_classes: int = 5,
    maxiter: int = 300,
    seed: int = 42,
    callback=None,
    simulator: str = "statevector",
    encoding: str = "angle",
    reuploads: int = 1,
    loss: str = "squared_error",
):
    """Return (classifier, circuit) ready for classifier.fit(X, y) with integer labels.

    simulator="statevector" uses the fast exact StatevectorQNN above;
    simulator="estimator" uses qiskit-machine-learning's EstimatorQNN with
    Qiskit's exact reference Estimator (same results, much slower).
    """
    algorithm_globals.random_seed = seed
    qnn, circuit = build_qnn(num_qubits, num_classes, encoding, reuploads, loss, simulator)
    classifier = NeuralNetworkClassifier(
        qnn,
        loss=loss,
        one_hot=True,
        optimizer=COBYLA(maxiter=maxiter),
        callback=callback,
    )
    return classifier, circuit


def predict(classifier, x: np.ndarray) -> np.ndarray:
    """Predicted integer grades, shape (N,)."""
    return np.asarray(classifier.predict(x)).reshape(-1).astype(int)
