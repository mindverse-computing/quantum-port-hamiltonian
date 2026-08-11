"""
ibm — IBM Quantum hardware execution path for the Qiskit Q-pHNN reference.
=========================================================================

The CUDA-Q port (``../../cudaq``) runs the network model on an NVIDIA GPU up to
the single-device statevector / shot-based ceiling. IBM Quantum is **not** a
CUDA-Q backend (CUDA-Q targets IonQ / Quantinuum / IQM / OQC / Quantum Circuits),
so real-hardware execution goes through this Qiskit-native subpackage instead.

It reuses the existing Qiskit circuit builders in ``../network`` and
``../non_dissipative`` — the same gates, same observable — and adds only the
IBM Runtime submission, transpilation, and result-parsing layer.

Modules
-------
- ``connection``   : load credentials from ``.env`` and open a QiskitRuntimeService.
- ``sample_circuit``: a minimal Bell-state test job to verify the connection.
- ``runner``       : transpile + submit a QGNN energy circuit via EstimatorV2.

Nothing here is imported at package load beyond the module names, so importing
``ibm`` on a machine without ``qiskit-ibm-runtime`` does not fail until you call
into ``connection``/``runner``.
"""

__all__ = ["connection", "sample_circuit", "runner"]
