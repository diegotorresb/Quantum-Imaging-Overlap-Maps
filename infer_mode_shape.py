import numpy as np
import pandas as pd
from itertools import product as iproduct

from forward_model import forward_model
from residual import residual


def run_task_A(O_obs, grid, sigma_x, sigma_y, rotation_angle=0.0,
               kmax=5, top_n=5, loss='ncc'):
    """
    Task A: infer mechanical mode integers (kx, ky) and beta, given beam parameters.

    Step 1 — brute-force single-mode scan over all (kx, ky) in [1..kmax]².
    Step 2 — two-mode linear solve (lstsq for beta) over the top-N shortlist.

    Returns dict: df, top1, O_model, params.
    """
    sx, sy, theta = sigma_x, sigma_y, rotation_angle
    O_obs_flat = O_obs.ravel()

    # Step 1: single-mode scan
    single_candidates = []
    for kx, ky in iproduct(range(1, kmax + 1), range(1, kmax + 1)):
        params = dict(kx1=kx, ky1=ky, kx2=kx, ky2=ky, beta=0.0,
                      sigma_x=sx, sigma_y=sy, rotation_angle=theta)
        O1 = forward_model(params, grid)
        score = residual(O1, O_obs, mode='ncc')
        single_candidates.append({'kx': kx, 'ky': ky, 'score': score, 'O': O1})

    # Sort by |NCC| so that anti-correlated modes (negative NCC) are also shortlisted
    # as potential secondary mode candidates in the two-mode solve.
    single_candidates.sort(key=lambda c: abs(c['score']), reverse=True)
    top_singles = single_candidates[:top_n]

    # Step 2: two-mode linear solve for all ordered pairs in the top-N shortlist
    two_mode_results = []
    for i, c1 in enumerate(top_singles):
        for j, c2 in enumerate(top_singles):
            if i == j:
                continue
            O1 = c1['O']
            O2 = c2['O']
            A_mat = np.column_stack([O1.ravel(), O2.ravel()])
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, O_obs_flat, rcond=None)
            scale_1, scale_2 = float(coeffs[0]), float(coeffs[1])
            # Ensure c1 is the dominant mode; swap if the solver placed more weight on c2.
            if abs(scale_2) > abs(scale_1):
                c1, c2 = c2, c1
                O1, O2 = O2, O1
                scale_1, scale_2 = scale_2, scale_1
            if abs(scale_1) < 1e-30:
                continue
            beta_fit = scale_2 / scale_1
            O_combined = scale_1 * O1 + scale_2 * O2
            ncc_score = residual(O_combined, O_obs, mode='ncc')
            rmse = float(np.sqrt(residual(O_combined, O_obs, mode='mse', normalize=True)))
            two_mode_results.append({
                'kx1': c1['kx'], 'ky1': c1['ky'],
                'kx2': c2['kx'], 'ky2': c2['ky'],
                'beta': beta_fit,
                'ncc_score': float(ncc_score),
                'rmse': float(rmse),
            })

    # Primary sort: NCC descending. Tie-break: |beta| ascending (smaller beta = more dominant primary).
    two_mode_results.sort(key=lambda r: (r['ncc_score'], -abs(r['beta'])), reverse=True)

    # Fallback to single-mode if no valid pairs were found
    if not two_mode_results:
        best = top_singles[0]
        params_best = dict(kx1=best['kx'], ky1=best['ky'],
                           kx2=best['kx'], ky2=best['ky'], beta=0.0,
                           sigma_x=sx, sigma_y=sy, rotation_angle=theta)
        return {
            'df': pd.DataFrame([{
                'kx1': c['kx'], 'ky1': c['ky'], 'kx2': c['kx'], 'ky2': c['ky'],
                'beta': 0.0, 'ncc_score': c['score'], 'rmse': 0.0
            } for c in top_singles]),
            'top1': {'kx1': best['kx'], 'ky1': best['ky'],
                     'kx2': best['kx'], 'ky2': best['ky'],
                     'beta': 0.0, 'ncc_score': float(best['score']), 'rmse': 0.0},
            'O_model': best['O'],
            'params': params_best,
        }

    best = two_mode_results[0]
    params_best = dict(kx1=best['kx1'], ky1=best['ky1'],
                       kx2=best['kx2'], ky2=best['ky2'],
                       beta=best['beta'], sigma_x=sx, sigma_y=sy, rotation_angle=theta)
    O_model_best = forward_model(params_best, grid)

    return {
        'df': pd.DataFrame(two_mode_results),
        'top1': best,
        'O_model': O_model_best,
        'params': params_best,
    }
