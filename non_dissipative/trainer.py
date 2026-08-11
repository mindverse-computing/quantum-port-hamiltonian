"""
non_dissipative/trainer.py
===========================
Training loop for the Quantum HNN (conservative systems).

Uses scipy BFGS optimiser — appropriate because StatevectorEstimator provides
exact (noiseless) expectation values, giving a smooth loss landscape.

Workflow
--------
1. Generate phase-space dataset (VectorFieldDataset)
2. Initialise QuantumHNN weights
3. Minimise vector-field MSE loss via BFGS
4. Validate on held-out test points and rolled-out trajectory
5. Return TrainingResult with all metrics and learned weights
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .quantum_hnn import QuantumHNN
from .data_generator import VectorFieldDataset


@dataclass
class TrainingResult:
    """Results returned by train_qhnn()."""

    theta_opt: np.ndarray
    """Optimal circuit weight vector."""

    final_loss: float
    """Training loss at convergence."""

    loss_history: list[float]
    """Loss value recorded at each optimiser iteration."""

    n_iter: int
    """Number of optimiser iterations."""

    train_q_dot_mse: float
    """MSE on training vector field, q̇ component."""

    train_p_dot_mse: float
    """MSE on training vector field, ṗ component."""

    test_q_dot_mse: float
    """MSE on held-out test set, q̇ component."""

    test_p_dot_mse: float
    """MSE on held-out test set, ṗ component."""

    wall_time_s: float = 0.0
    """Total wall-clock time in seconds."""

    optimizer_result: OptimizeResult | None = None
    """Full scipy OptimizeResult object."""


def evaluate_vector_field_mse(
    model: QuantumHNN,
    phi: np.ndarray,
    dataset: VectorFieldDataset,
) -> tuple[float, float]:
    """
    Compute separate MSE for q̇ and ṗ on a dataset.

    phi = [θ₀…θ_{n-1}, s, b] (full parameter vector with scale and offset).
    Returns (q_dot_mse, p_dot_mse).
    """
    q_dot_errs, p_dot_errs = [], []
    for i in range(dataset.n_samples):
        q_d_pred = model.q_dot(dataset.q[i], dataset.p[i], phi)
        p_d_pred = model.p_dot(dataset.q[i], dataset.p[i], phi)
        q_dot_errs.append((q_d_pred - dataset.q_dot[i])**2)
        p_dot_errs.append((p_d_pred - dataset.p_dot[i])**2)
    return float(np.mean(q_dot_errs)), float(np.mean(p_dot_errs))


def train_qhnn(
    model: QuantumHNN,
    train_data: VectorFieldDataset,
    test_data: VectorFieldDataset | None = None,
    max_iter: int = 100,
    verbose: bool = True,
) -> TrainingResult:
    """
    Train a QuantumHNN on a vector-field dataset using BFGS.

    Parameters
    ----------
    model : QuantumHNN
        Untrained model (will be modified in-place: theta_opt and loss_history).
    train_data : VectorFieldDataset
        Training phase-space samples and true vector field.
    test_data : VectorFieldDataset, optional
        Held-out test set for generalisation evaluation.
    max_iter : int
        Maximum BFGS iterations.
    verbose : bool
        Print progress every 10 iterations.

    Returns
    -------
    TrainingResult
    """
    iteration_counter = [0]
    loss_history: list[float] = []

    q_data = train_data.q
    p_data = train_data.p
    q_dot_true = train_data.q_dot
    p_dot_true = train_data.p_dot

    def objective(phi: np.ndarray) -> float:
        loss = model.compute_loss(phi, q_data, p_data, q_dot_true, p_dot_true)
        loss_history.append(loss)
        iteration_counter[0] += 1
        if verbose and iteration_counter[0] % 10 == 0:
            s = phi[-2]
            print(f"  [Q-HNN] Iter {iteration_counter[0]:4d} | loss = {loss:.6f} | s = {s:.4f}")
        return loss

    phi_init = model.init_weights()
    t0 = time.time()

    if verbose:
        print(f"[Q-HNN] Training on {train_data.n_samples} samples | "
              f"max_iter={max_iter}")
        print(f"  System: {train_data.name}")
        print(f"  Initial phi (theta | s | b): {np.round(phi_init, 4)}")

    opt_result = minimize(
        objective,
        phi_init,
        method="BFGS",
        options={"maxiter": max_iter, "disp": False},
    )

    wall_time = time.time() - t0
    phi_opt = opt_result.x

    # --- Post-training sign correction -----------------------------------
    # If BFGS converged to -H instead of +H, negate s to flip vector field.
    phi_opt = model.apply_sign_correction(
        phi_opt, q_data, p_data, q_dot_true, p_dot_true
    )
    if verbose:
        print(f"  Scale s after sign-correction: {phi_opt[-2]:.4f}")

    model.theta_opt = phi_opt
    model.loss_history = loss_history

    if verbose:
        print(f"\n[Q-HNN] Optimization complete in {wall_time:.1f}s")
        print(f"  Final loss:      {opt_result.fun:.6f}")
        print(f"  Iterations:      {opt_result.nit}")
        print(f"  Converged:       {opt_result.success}")
        print(f"  Learned phi (theta | s | b): {np.round(phi_opt, 4)}")

    # --- Component-wise MSE on training set ---
    train_q_mse, train_p_mse = evaluate_vector_field_mse(model, phi_opt, train_data)

    # --- Test set evaluation ---
    test_q_mse, test_p_mse = 0.0, 0.0
    if test_data is not None:
        test_q_mse, test_p_mse = evaluate_vector_field_mse(model, phi_opt, test_data)
        if verbose:
            print(f"\n[Q-HNN] Test evaluation ({test_data.n_samples} samples):")
            print(f"  Test q̇ MSE: {test_q_mse:.6f}")
            print(f"  Test ṗ MSE: {test_p_mse:.6f}")

    return TrainingResult(
        theta_opt=phi_opt,
        final_loss=float(opt_result.fun),
        loss_history=loss_history,
        n_iter=opt_result.nit,
        train_q_dot_mse=train_q_mse,
        train_p_dot_mse=train_p_mse,
        test_q_dot_mse=test_q_mse,
        test_p_dot_mse=test_p_mse,
        wall_time_s=wall_time,
        optimizer_result=opt_result,
    )
