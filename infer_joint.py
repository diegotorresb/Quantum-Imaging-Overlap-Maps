import numpy as np

from infer_mode_shape import run_task_A
from infer_illumination import run_task_B


def infer_joint(O_obs, grid, kmax=5, max_outer=10, tol=1e-4):
    """
    Task C: alternating minimization for the fully blind case.

    Alternates Task A (mode shape, fast linear solve) and Task B (beam, nonlinear)
    until ΔNCC < tol or max_outer iterations.
    """
    L = grid['L']
    sigma0 = L / 20

    result_A = run_task_A(O_obs, grid, sigma_x=sigma0, sigma_y=sigma0,
                          rotation_angle=0.0, kmax=kmax, top_n=1)
    mode_params = {k: result_A['top1'][k] for k in ['kx1', 'ky1', 'kx2', 'ky2', 'beta']}
    beam_params = {'sigma_x': sigma0, 'sigma_y': sigma0, 'rotation_angle': 0.0}
    ncc_prev = -np.inf

    for _ in range(max_outer):
        result_B = run_task_B(O_obs, grid, **mode_params)
        beam_params = {k: result_B[k] for k in ['sigma_x', 'sigma_y', 'rotation_angle']}

        result_A = run_task_A(O_obs, grid,
                              sigma_x=beam_params['sigma_x'],
                              sigma_y=beam_params['sigma_y'],
                              rotation_angle=beam_params['rotation_angle'],
                              kmax=kmax, top_n=1)
        mode_params = {k: result_A['top1'][k] for k in ['kx1', 'ky1', 'kx2', 'ky2', 'beta']}
        ncc_curr = float(result_A['top1']['ncc_score'])

        if abs(ncc_curr - ncc_prev) < tol:
            break
        ncc_prev = ncc_curr

    return {
        'mode_params': mode_params,
        'beam_params': beam_params,
        'O_model': result_A['O_model'],
        'params': {**mode_params, **beam_params},
        'ncc_score': float(result_A['top1']['ncc_score']),
    }
