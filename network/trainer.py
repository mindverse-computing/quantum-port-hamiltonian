"""
network/trainer.py
==================
BFGS vector-field trainer for the NetworkQpHNN.

Learns the QGNN circuit weights (and, in dissipative gamma mode, the per-node
damping vector γ) by minimising the network port-Hamiltonian vector-field MSE

    L(params) = (1/B) Σ_b ‖ ẋ_pred(x_b) − ẋ_true(x_b) ‖² .

Uses scipy BFGS — appropriate because StatevectorEstimator gives exact,
noiseless expectation values, so the loss landscape is smooth. Returns a
NetworkTrainingResult with per-node metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .quantum_network_phnn import NetworkQpHNN, _softplus
from .data_generator import NetworkVectorFieldDataset


@dataclass
class NetworkTrainingResult:
    params_opt: np.ndarray
    final_loss: float
    loss_history: list[float]
    n_iter: int
    train_mse: float
    test_mse: float = float("nan")
    phi_dot_mse: float = float("nan")
    omega_dot_mse: float = float("nan")
    learned_gamma: np.ndarray | None = None
    true_gamma: np.ndarray | None = None
    gamma_rel_err: float = float("nan")
    wall_time_s: float = 0.0
    optimizer_result: OptimizeResult | None = None


def _component_mse(
    model: NetworkQpHNN, params: np.ndarray, ds: NetworkVectorFieldDataset,
) -> tuple[float, float]:
    """Return (phi_dot_mse, omega_dot_mse) over a dataset."""
    N = model.N
    perr, oerr = [], []
    for b in range(ds.n_samples):
        pred = model.vector_field(ds.states[b], params)
        true = ds.d_states[b]
        perr.append(np.mean((pred[:N] - true[:N]) ** 2))
        oerr.append(np.mean((pred[N:] - true[N:]) ** 2))
    return float(np.mean(perr)), float(np.mean(oerr))


def train_network_qphnn(
    model: NetworkQpHNN,
    train_data: NetworkVectorFieldDataset,
    test_data: NetworkVectorFieldDataset | None = None,
    max_iter: int = 100,
    verbose: bool = True,
) -> NetworkTrainingResult:
    """Train a NetworkQpHNN with BFGS on the vector-field MSE."""
    loss_history: list[float] = []
    it = [0]

    def objective(params: np.ndarray) -> float:
        loss = model.compute_loss(params, train_data.states, train_data.d_states)
        loss_history.append(loss)
        it[0] += 1
        if verbose and it[0] % 10 == 0:
            msg = f"  [Net-QpHNN] iter {it[0]:4d} | loss = {loss:.6f}"
            if model.dissipative and model.diss_mode == "gamma":
                g = _softplus(params[model.n_circuit_weights:])
                msg += f" | γ̄ = {g.mean():.3f}"
            print(msg)
        return loss

    p0 = model.init_params()
    t0 = time.time()
    if verbose:
        print(f"[Net-QpHNN] Training {model} on {train_data.n_samples} samples "
              f"(max_iter={max_iter})")
        print(f"  System: {train_data.name} | N={model.N} | "
              f"edges={len(model.edges)}")

    res = minimize(objective, p0, method="BFGS",
                   options={"maxiter": max_iter, "disp": False})

    wall = time.time() - t0
    model.params_opt = res.x
    model.loss_history = loss_history

    train_mse = float(res.fun)
    pmse, omse = _component_mse(model, res.x, train_data)
    test_mse = float("nan")
    if test_data is not None:
        test_mse = model.compute_loss(res.x, test_data.states, test_data.d_states)

    learned_gamma = None
    gamma_rel = float("nan")
    if model.dissipative and model.diss_mode == "gamma":
        learned_gamma = _softplus(res.x[model.n_circuit_weights:])
        if train_data.gamma is not None and np.any(train_data.gamma > 0):
            gamma_rel = float(
                np.mean(np.abs(learned_gamma - train_data.gamma)
                        / (np.abs(train_data.gamma) + 1e-9))
            )

    if verbose:
        print(f"\n[Net-QpHNN] Done in {wall:.1f}s | final loss {train_mse:.6f} "
              f"| iters {res.nit}")
        print(f"  φ̇ MSE = {pmse:.6f} | ω̇ MSE = {omse:.6f}")
        if learned_gamma is not None:
            print(f"  learned γ = {np.round(learned_gamma, 3)}")
            print(f"  true    γ = {np.round(train_data.gamma, 3)}")
            print(f"  mean |Δγ|/γ = {gamma_rel*100:.2f}%")

    return NetworkTrainingResult(
        params_opt=res.x, final_loss=train_mse, loss_history=loss_history,
        n_iter=res.nit, train_mse=train_mse, test_mse=test_mse,
        phi_dot_mse=pmse, omega_dot_mse=omse,
        learned_gamma=learned_gamma, true_gamma=train_data.gamma,
        gamma_rel_err=gamma_rel, wall_time_s=wall, optimizer_result=res,
    )
