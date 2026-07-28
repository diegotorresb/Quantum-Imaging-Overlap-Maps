import numpy as np
from scipy.optimize import differential_evolution, minimize

from forward_model import forward_model
from residual import residual


def run_task_B(O_obs, grid, kx1=1, ky1=1, kx2=1, ky2=1, beta=0.0,
               isotropic=False, loss='ncc',
               sigma_min=0.001, sigma_max=1.0):
    """
    Task B: infer beam parameters (sigma_x, sigma_y, rotation_angle), given mechanical mode.

    Uses differential evolution (global) then Nelder-Mead (local) to maximize NCC.
    Returns dict with beam params, 1-sigma uncertainties, NCC, RMSE, O_model, params.
    """
    def _params(p):
        if isotropic:
            return dict(kx1=kx1, ky1=ky1, kx2=kx2, ky2=ky2, beta=beta,
                        sigma_x=float(p[0]), sigma_y=float(p[0]), rotation_angle=float(p[1]))
        return dict(kx1=kx1, ky1=ky1, kx2=kx2, ky2=ky2, beta=beta,
                    sigma_x=float(p[0]), sigma_y=float(p[1]), rotation_angle=float(p[2]))

    def objective(p):
        O_model = forward_model(_params(p), grid)
        return -residual(O_model, O_obs, mode='ncc')

    # Step 1: moment-based sigma estimate as initialization hint
    O_abs = np.abs(O_obs)
    total = float(O_abs.sum())
    if total > 0:
        mu_x = float(np.sum(grid['X'] * O_abs) / total)
        mu_y = float(np.sum(grid['Y'] * O_abs) / total)
        sigma_est = float(np.sqrt(
            np.sum(((grid['X'] - mu_x)**2 + (grid['Y'] - mu_y)**2) * O_abs) / total / 2
        ))
    else:
        sigma_est = grid['L'] / 20
    sigma_est = float(np.clip(sigma_est, sigma_min, sigma_max))

    # Step 2: global search then local refinement
    if isotropic:
        bounds = [(sigma_min, sigma_max), (-90.0, 90.0)]
    else:
        bounds = [(sigma_min, sigma_max), (sigma_min, sigma_max), (-90.0, 90.0)]

    result_global = differential_evolution(
        objective, bounds, maxiter=300, tol=1e-5, seed=42, workers=-1
    )
    result_local = minimize(
        objective, result_global.x,
        method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 5000}
    )

    p_opt = result_local.x
    params_opt = _params(p_opt)
    O_model_opt = forward_model(params_opt, grid)
    ncc_score = residual(O_model_opt, O_obs, mode='ncc')
    rmse = float(np.sqrt(residual(O_model_opt, O_obs, mode='mse', normalize=True)))

    # Step 3: Jacobian-based parameter uncertainties via central differences
    try:
        eps = 1e-6
        J = np.zeros((O_obs.size, len(p_opt)))
        for i in range(len(p_opt)):
            p_plus, p_minus = p_opt.copy(), p_opt.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            J[:, i] = (
                forward_model(_params(p_plus), grid).ravel()
                - forward_model(_params(p_minus), grid).ravel()
            ) / (2 * eps)
        sigma_r2 = float(np.var(O_model_opt.ravel() - O_obs.ravel()))
        JtJ = J.T @ J
        Cov = sigma_r2 * np.linalg.inv(JtJ)
        uncertainties = np.sqrt(np.abs(np.diag(Cov)))
    except (np.linalg.LinAlgError, Exception):
        uncertainties = np.full(len(p_opt), np.nan)

    if isotropic:
        result = {
            'sigma_x': float(p_opt[0]), 'sigma_y': float(p_opt[0]),
            'rotation_angle': float(p_opt[1]),
            'sigma_x_err': float(uncertainties[0]), 'sigma_y_err': float(uncertainties[0]),
            'rotation_angle_err': float(uncertainties[1]),
        }
    else:
        result = {
            'sigma_x': float(p_opt[0]), 'sigma_y': float(p_opt[1]),
            'rotation_angle': float(p_opt[2]),
            'sigma_x_err': float(uncertainties[0]), 'sigma_y_err': float(uncertainties[1]),
            'rotation_angle_err': float(uncertainties[2]),
        }

    result.update({'ncc_score': float(ncc_score), 'rmse': rmse,
                   'O_model': O_model_opt, 'params': params_opt})
    return result
