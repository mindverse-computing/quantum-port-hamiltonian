"""
network/ — N-node network quantum port-Hamiltonian package.

Topology-entangled QGNN energy surrogate + network quantum port-Hamiltonian
model with multi-ancilla MINL dissipation. See theory/network_qphnn_theory.md.
"""
from .qgnn_energy import QGNNEnergy, edges_from_coupling
from .quantum_network_phnn import NetworkQpHNN
from .data_generator import (
    gen_network_conservative, gen_network_dissipative,
    build_ring_coupling, build_modular_coupling,
    build_star_coupling, build_chain_coupling,
    NetworkVectorFieldDataset,
)
from .trainer import train_network_qphnn, NetworkTrainingResult
