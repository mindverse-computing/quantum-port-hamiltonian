"""
dissipative/trainer.py
=======================
Training loops for both Q-pHNN variants.

train_dynamic_qphnn()
    Trains DynamicQpHNN (v1) using COBYLA — gradient-free, suitable for
    shot-noisy AerSimulator circuits with mid-circuit measurements.

train_vector_field_qphnn()
    Trains VectorFieldQpHNN (v2) using BFGS — exact gradients via
    StatevectorEstimator, learns energy manifold + damping coefficient.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .quantum_phnn import DynamicQpHNN, VectorFieldQpHNN
from .data_generator import DissipativeVectorFieldDataset, TrajectoryDataset


# ===========================================================================
# Shared result container
# ===========================================================================

@dataclass
class DissipativeTrainingResult:
    """Results from either Q-pHNN training procedure."""

    params_opt: np.ndarray
    """Optimal parameter vector (θ for v1; [θ, γ] for v2)."""

    final_loss: float
    """Loss at convergence."""

    loss_history: list[float]
    """Loss at each function evaluation."""

    n_iter: int
    """Optimiser iterations."""

    wall_time_s: float = 0.0
    """Total training time in seconds."""

    learned_gamma: float | None = None
    """Learned damping coefficient (v2 only)."""

    true_gamma: float | None = None
    """Ground-truth damping (for error reporting)."""

    optimizer_result: OptimizeResult | None = None


# ===========================================================================
# v1 trainer — COBYLA, shot-based trajectory fitting
# ===========================================================================

def train_dynamic_qphnn(
    model: DynamicQpHNN,
    trajectory: TrajectoryDataset,
    init_angle: float | None = None,
    max_iter: int = 80,
    rhobeg: float = 0.5,
    n_shots: int = 30,      # renamed from train_shots; matches DynamicQpHNN.compute_loss
    verbose: bool = True,
) -> DissipativeTrainingResult:
    """
    Train DynamicQpHNN (v1) using COBYLA.

    Fits [θ_J, θ_R, θ_kick] to minimise MSE between the quantum circuit's
    <σ_x> trajectory and the ground-truth q(t) trajectory.

    Parameters
    ----------
    model : DynamicQpHNN
    trajectory : TrajectoryDataset
        Integrated classical trajectory (t, q, p).
    init_angle : float, optional
        Initial state preparation angle (default: π/2).
    max_iter : int
        COBYLA maximum function evaluations.
    rhobeg : float
        COBYLA initial step size.
    train_shots : int
        Shot count during training (lower = faster but noisier).
    verbose : bool

    Returns
    -------
    DissipativeTrainingResult
    """
    if init_angle is None:
        init_angle = np.pi / 2.0

    # Target: q values at discrete time steps (skip t=0)
    target_q = list(trajectory.q[1:model.n_steps + 1])
    n_pts = len(target_q)

    loss_history: list[float] = []
    iter_count = [0]

    def objective(params: np.ndarray) -> float:
        loss = model.compute_loss(
            params, target_q, init_angle, n_shots=n_shots
        )
        loss_history.append(loss)
        iter_count[0] += 1
        if verbose and iter_count[0] % 10 == 0:
            print(f"  [Q-pHNN v1] Iter {iter_count[0]:4d} | loss = {loss:.6f}")
        return loss

    initial_params = np.array([0.1, 0.1, 0.0])

    t0 = time.time()
    if verbose:
        print(f"[Q-pHNN v1] Training on {n_pts}-step trajectory | "
              f"max_iter={max_iter}, shots={n_shots}")
        print(f"  Target q: {np.round(target_q, 3)}")

    opt_result = minimize(
        objective,
        initial_params,
        method="COBYLA",
        options={"maxiter": max_iter, "rhobeg": rhobeg},
    )

    wall_time = time.time() - t0
    params_opt = opt_result.x
    model.params_opt = params_opt
    model.loss_history = loss_history

    if verbose:
        print(f"\n[Q-pHNN v1] Done in {wall_time:.1f}s")
        print(f"  θ_J (conservative) = {params_opt[0]:.4f}")
        print(f"  θ_R (dissipative)  = {params_opt[1]:.4f}")
        print(f"  θ_kick (nonlinear) = {params_opt[2]:.4f}")
        print(f"  Final loss:        {opt_result.fun:.6f}")

        # Predict final trajectory for display (returns q_arr, p_arr)
        pred_q, _ = model.predict_trajectory(params_opt, init_angle, n_shots=100)
        print(f"  Target:    {np.round(target_q, 3)}")
        print(f"  Predicted: {np.round(pred_q[:n_pts], 3)}")

    return DissipativeTrainingResult(
        params_opt=params_opt,
        final_loss=float(opt_result.fun),
        loss_history=loss_history,
        n_iter=opt_result.nfev,
        wall_time_s=wall_time,
        true_gamma=trajectory.true_gamma,
        optimizer_result=opt_result,
    )


# ===========================================================================
# v2 trainer — BFGS, exact gradient, vector-field fitting
# ===========================================================================

def evaluate_dissipative_mse(
    model: VectorFieldQpHNN,
    params: np.ndarray,
    dataset: DissipativeVectorFieldDataset,
) -> tuple[float, float, float]:
    """
    Compute separate q̇, ṗ MSE and learned damping error.
    params = [θ₀…θ_{n-1}, s, γ]
    Returns (q_dot_mse, p_dot_mse, gamma_error).
    """
    theta = params[: model.n_circuit_weights]
    s     = float(params[-2])
    gamma = float(params[-1])

    q_errs, p_errs = [], []
    for i in range(dataset.n_samples):
        q_d = model.q_dot(dataset.q[i], dataset.p[i], theta, s)
        p_d = model.p_dot(dataset.q[i], dataset.p[i], theta, gamma, s)
        q_errs.append((q_d - dataset.q_dot[i])**2)
        p_errs.append((p_d - dataset.p_dot[i])**2)

    gamma_err = abs(gamma - dataset.true_gamma) if dataset.true_gamma else float("nan")
    return float(np.mean(q_errs)), float(np.mean(p_errs)), gamma_err


def train_vector_field_qphnn(
    model: VectorFieldQpHNN,
    train_data: DissipativeVectorFieldDataset,
    test_data: DissipativeVectorFieldDataset | None = None,
    max_iter: int = 100,
    lambda_passivity: float = 0.0,
    verbose: bool = True,
) -> DissipativeTrainingResult:
    """
    Train VectorFieldQpHNN (v2) using BFGS.

    Learns [θ₀..θ_{n-1}, γ] jointly to minimise the port-Hamiltonian
    vector-field MSE:
        L = ‖q̇_pred − q̇_true‖² + ‖ṗ_pred − ṗ_true‖²

    Optionally adds a passivity penalty (lambda_passivity > 0) that
    penalises positive energy rates Ḣ > 0 over the training distribution,
    pushing the energy-monotone fraction toward 100%.

    Parameters
    ----------
    model : VectorFieldQpHNN
    train_data : DissipativeVectorFieldDataset
    test_data : DissipativeVectorFieldDataset, optional
    max_iter : int
    lambda_passivity : float
        Weight for the passivity penalty term. 0 = pure vector-field MSE.
        0.05–0.2 is a good range to enforce Ḣ ≤ 0.
    verbose : bool
    """
    loss_history: list[float] = []
    iter_count = [0]

    def objective(params: np.ndarray) -> float:
        if lambda_passivity > 0.0:
            loss = model.compute_loss_with_passivity(
                params,
                train_data.q, train_data.p,
                train_data.q_dot, train_data.p_dot,
                lambda_passivity=lambda_passivity,
            )
        else:
            loss = model.compute_loss(
                params,
                train_data.q, train_data.p,
                train_data.q_dot, train_data.p_dot,
            )
        loss_history.append(loss)
        iter_count[0] += 1
        if verbose and iter_count[0] % 10 == 0:
            s_now     = params[-2]
            gamma_now = params[-1]
            print(f"  [Q-pHNN v2] Iter {iter_count[0]:4d} | "
                  f"loss = {loss:.6f} | s = {s_now:.4f} | γ = {gamma_now:.4f}")
        return loss

    params_init = model.init_params()
    t0 = time.time()

    if verbose:
        print(f"[Q-pHNN v2] Training on {train_data.n_samples} samples | "
              f"max_iter={max_iter}, λ_passivity={lambda_passivity}")
        print(f"  System: {train_data.name}")
        print(f"  True damping γ = {train_data.true_gamma}")
        print(f"  Initial params: θ={np.round(params_init[:-2], 3)}, s={params_init[-2]:.2f}, γ={params_init[-1]:.3f}")

    opt_result = minimize(
        objective,
        params_init,
        method="BFGS",
        options={"maxiter": max_iter, "disp": False},
    )

    wall_time = time.time() - t0
    params_opt = opt_result.x

    # --- Post-training sign correction ---
    params_opt = model.apply_sign_correction(
        params_opt, train_data.q, train_data.p,
        train_data.q_dot, train_data.p_dot,
    )

    model.params_opt = params_opt
    model.loss_history = loss_history

    s_opt     = params_opt[-2]
    gamma_opt = params_opt[-1]

    if verbose:
        print(f"\n[Q-pHNN v2] Done in {wall_time:.1f}s")
        print(f"  Final loss:     {opt_result.fun:.6f}")
        print(f"  Iterations:     {opt_result.nit}")
        print(f"  Converged:      {opt_result.success}")
        print(f"  Learned s:      {s_opt:.4f}")
        print(f"  Learned γ:      {gamma_opt:.4f}")
        if train_data.true_gamma is not None:
            print(f"  True γ:         {train_data.true_gamma:.4f}")
            print(f"  |Δγ|:           {abs(gamma_opt - train_data.true_gamma):.4f}")

    # --- Component-wise MSE ---
    train_q_mse, train_p_mse, gamma_err = evaluate_dissipative_mse(
        model, params_opt, train_data
    )

    if verbose:
        print(f"\n  Train q̇ MSE: {train_q_mse:.6f}")
        print(f"  Train ṗ MSE: {train_p_mse:.6f}")

    if test_data is not None:
        test_q_mse, test_p_mse, _ = evaluate_dissipative_mse(
            model, params_opt, test_data
        )
        if verbose:
            print(f"\n  Test q̇ MSE:  {test_q_mse:.6f}")
            print(f"  Test ṗ MSE:  {test_p_mse:.6f}")

    return DissipativeTrainingResult(
        params_opt=params_opt,
        final_loss=float(opt_result.fun),
        loss_history=loss_history,
        n_iter=opt_result.nit,
        wall_time_s=wall_time,
        learned_gamma=float(gamma_opt),
        true_gamma=train_data.true_gamma,
        optimizer_result=opt_result,
    )
