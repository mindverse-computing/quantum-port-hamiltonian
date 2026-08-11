"""
ibm/minl_dynamic.py
===================
Measurement-Induced NonLinearity as a **native IBM dynamic circuit**.

The reference implementation in ``network.quantum_network_phnn`` realises MINL
with ``Statevector.measure`` plus a Python ``if`` — correct, but a simulation of
the channel. On hardware the same channel is expressed directly: a mid-circuit
``measure`` followed by an ``if_test`` block whose body runs conditioned on the
outcome, inside the qubits' coherence time. This is the construct IBM exposes
for classical feedforward, and it is what makes the dissipation physical rather
than post-processed: energy leaves the system through an actual measurement on
the device.

Support is not universal — older control electronics cannot do it. Check the
target before submitting::

    supports_dynamic(backend)          # -> True on ibm_marrakesh (Heron r2)

One Trotter block, mirroring ``NetworkQpHNN.run_minl_trajectory`` gate for gate:

    U_J   : Rz(theta_J_i) on each node, Rzz(theta_J) on each edge  (conservative)
    U_R   : CRy(theta_R_i) from node i to its bath ancilla         (system-bath)
    MINL  : measure ancilla -> if 1: Rx(theta_k_i) on node i -> reset ancilla

**Ancilla reuse.** Measuring and resetting lets one ancilla be time-multiplexed
across all nodes within a step, so the register is ``N + 1`` qubits rather than
``2N``. That is what puts large N on a 156-qubit chip, at the cost of serialised
measurement rounds (depth for width). Set ``ancilla_reuse=False`` to get the
one-ancilla-per-node layout instead.

Validated offline: against the statevector reference at N=3, 3 steps, the
dynamic circuit reproduces per-node <X_i> to within sampling error.

Note the deprecations that shape this design — ``while``/``for``/``switch``,
nested control flow, and conditional measurements were all withdrawn ahead of
IBM's newer dynamic-circuit stack, so this module uses a flat ``if_test`` with
an unconditional body and nothing nested inside it.
"""
from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def supports_dynamic(backend) -> bool:
    """True when *backend* accepts mid-circuit measurement with feedforward."""
    try:
        if "dynamic_circuits" in backend.configuration().supported_features:
            return True
    except Exception:                                          # noqa: BLE001
        pass
    try:
        return "if_else" in backend.target.operation_names
    except Exception:                                          # noqa: BLE001
        return False


def minl_dynamic_circuit(N: int, edges, theta_J, theta_R, theta_k, state0,
                         steps: int, *, ancilla_reuse: bool = True,
                         readout: str = "x") -> QuantumCircuit:
    """
    Build the MINL dissipative channel as a dynamic circuit.

    Parameters
    ----------
    state0 : array (2N,)
        Phase-space point; first N entries are positions (Rx), last N momenta
        (Ry). Momentum must use Ry — Rz commutes with the X read-out and would
        leave the momentum coordinate invisible.
    readout : {"x", "z"}
        ``"x"`` measures <X_i>, the position observable the trajectory tracks.

    Returns a circuit on ``N + (1 if ancilla_reuse else N)`` qubits with one
    classical bit per (step, node) mid-circuit measurement plus an N-bit
    terminal register named ``out``.
    """
    theta_J = np.asarray(theta_J, float)
    theta_R = np.asarray(theta_R, float)
    theta_k = np.asarray(theta_k, float)
    state0 = np.asarray(state0, float)

    sys_r = QuantumRegister(N, "s")
    anc_r = QuantumRegister(1 if ancilla_reuse else N, "a")
    out = ClassicalRegister(N, "out")
    # A zero-step circuit is the t=0 point of a decay curve, and Runtime rejects
    # a zero-width classical register -- so only declare `m` when it is used.
    if steps > 0:
        mid = ClassicalRegister(steps * N, "m")
        qc = QuantumCircuit(sys_r, anc_r, mid, out)
    else:
        mid = None
        qc = QuantumCircuit(sys_r, anc_r, out)

    for i in range(N):                                  # data encoding
        qc.rx(float(state0[i]), sys_r[i])
        qc.ry(float(state0[N + i]), sys_r[i])
    qc.barrier()

    coupling = float(theta_J[N % len(theta_J)])
    for step in range(steps):
        for i in range(N):                              # U_J local phase
            qc.rz(float(theta_J[i]), sys_r[i])
        for (i, j) in edges:                            # U_J Ising coupling
            qc.rzz(coupling, sys_r[i], sys_r[j])
        for i in range(N):
            a = anc_r[0] if ancilla_reuse else anc_r[i]
            qc.cry(float(theta_R[i]), sys_r[i], a)      # entangle with bath
            cbit = mid[step * N + i]
            qc.measure(a, cbit)                         # Born-rule collapse
            with qc.if_test((cbit, 1)):                 # classical feedforward
                qc.rx(float(theta_k[i]), sys_r[i])
            qc.reset(a)                                 # reuse the ancilla
        qc.barrier()

    if readout == "x":
        for i in range(N):
            qc.h(sys_r[i])
    qc.measure(sys_r, out)
    return qc


def conservative_control_circuit(N: int, edges, theta_J, state0, steps: int,
                                 *, readout: str = "x") -> QuantumCircuit:
    """
    The same circuit with the MINL channel switched OFF (gamma = 0).

    Unitary by construction, so energy is conserved and any measured decay is
    device decoherence. This is the noise floor that must be subtracted before
    any decay in the MINL run can be called physical dissipation.
    """
    theta_J = np.asarray(theta_J, float)
    state0 = np.asarray(state0, float)
    sys_r = QuantumRegister(N, "s")
    out = ClassicalRegister(N, "out")
    qc = QuantumCircuit(sys_r, out)
    for i in range(N):
        qc.rx(float(state0[i]), sys_r[i])
        qc.ry(float(state0[N + i]), sys_r[i])
    qc.barrier()
    coupling = float(theta_J[N % len(theta_J)])
    for _ in range(steps):
        for i in range(N):
            qc.rz(float(theta_J[i]), sys_r[i])
        for (i, j) in edges:
            qc.rzz(coupling, sys_r[i], sys_r[j])
        qc.barrier()
    if readout == "x":
        for i in range(N):
            qc.h(sys_r[i])
    qc.measure(sys_r, out)
    return qc


def x_expectations(counts: dict, N: int) -> np.ndarray:
    """
    Per-node <X_i> from a counts dict produced by these circuits.

    Qiskit renders multiple classical registers space-separated, most recently
    declared first, so the terminal ``out`` register is the LEFT-most field.
    """
    total = sum(counts.values()) or 1
    ev = np.zeros(N)
    for bits, c in counts.items():
        out_field = bits.split()[0]
        rev = out_field[::-1]                 # bit i of the register
        for i in range(N):
            ev[i] += c * (1 - 2 * int(rev[i]))
    return ev / total


def phase_space_energy(counts: dict, N: int) -> float:
    """Scalar E_ps = sum_i <X_i>^2 -- the decaying quantity in the MINL run."""
    return float(np.sum(x_expectations(counts, N) ** 2))
