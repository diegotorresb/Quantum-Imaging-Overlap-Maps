import numpy as np
from physics_core import drum_mode_square, overlap_map


def forward_model(params: dict, grid: dict, mn: tuple = (0, 0)) -> np.ndarray:
    """
    params keys: beta, kx1, ky1, kx2, ky2, sigma_x, sigma_y, rotation_angle
    grid keys:   X, Y, dx, dy, A, L
    mn:          HG indices of signal mode (default u_00 illumination)
    """
    phi = drum_mode_square(
        grid['X'], grid['Y'], grid['L'],
        params['beta'], params['kx1'], params['ky1'],
        params['kx2'], params['ky2']
    ) * grid['A']
    m, n = mn
    return overlap_map(
        m, n, params['sigma_x'], params['sigma_y'],
        phi, grid['X'], grid['Y'], grid['dx'], grid['dy'],
        params['rotation_angle']
    )
