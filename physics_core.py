import numpy as np
from scipy.special import hermite, factorial
from scipy.signal import fftconvolve


def drum_mode_square(x, y, L, beta, kx1=1, ky1=1, kx2=1, ky2=1):
    return (np.sin(np.pi * kx1 * (x + L/2) / L) * np.sin(np.pi * ky1 * (y + L/2) / L)
            + beta * np.sin(np.pi * ky2 * (y + L/2) / L) * np.sin(np.pi * kx2 * (x + L/2) / L))


def HG_1D(x, n, sigma):
    Hn = hermite(n)
    xi = x / sigma
    gauss = (1 / (np.pi * sigma**2))**0.25 * np.exp(-x**2 / (2 * sigma**2))
    return gauss * Hn(xi) / np.sqrt(2**n * factorial(n))


def HG_2D(x, y, n, m, sigma_x, sigma_y):
    return HG_1D(x, n, sigma_x) * HG_1D(y, m, sigma_y)


def rotated_HG_2D(X, Y, n, m, sigma_x, sigma_y, rotation_angle=0):
    cos_theta = np.cos(np.radians(rotation_angle))
    sin_theta = np.sin(np.radians(rotation_angle))
    x_rot = X * cos_theta + Y * sin_theta
    y_rot = -X * sin_theta + Y * cos_theta
    return HG_2D(x_rot, y_rot, n, m, sigma_x, sigma_y)


def optical_fields_product(X, Y, m, n, sigx, sigy, x0=0.0, y0=0.0, rotation_angle=0, rel_strength=1):
    u_mn = rotated_HG_2D(X - x0, Y - y0, n, m, sigx, sigy, rotation_angle)
    u_00 = rel_strength * rotated_HG_2D(X - x0, Y - y0, 0, 0, sigx, sigy, rotation_angle)
    return u_mn * u_00


def overlap_map(m, n, sigx, sigy, phi, X, Y, dx, dy, rotation_angle=0, rel_strength=1):
    u_mn_0 = rotated_HG_2D(X, Y, n, m, sigx, sigy, rotation_angle)
    u_00_0 = rel_strength * rotated_HG_2D(X, Y, 0, 0, sigx, sigy, rotation_angle)
    g0 = u_mn_0 * u_00_0
    return fftconvolve(phi, g0[::-1, ::-1], mode='same') * dx * dy


def make_grid(L, N, chip_margin=0.0):
    """Returns grid dict with x, y, X, Y, dx, dy, A, L. A is the membrane aperture mask."""
    L_total = L + 2 * chip_margin
    x = np.linspace(-L_total / 2, L_total / 2, N)
    y = np.linspace(-L_total / 2, L_total / 2, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='xy')
    A = ((np.abs(X) <= L / 2) & (np.abs(Y) <= L / 2)).astype(float)
    return {'x': x, 'y': y, 'X': X, 'Y': Y, 'dx': dx, 'dy': dy, 'A': A, 'L': L}


def make_grid_from_arrays(x, y, L):
    """Build grid dict from coordinate arrays."""
    X, Y = np.meshgrid(x, y, indexing='xy')
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    A = ((np.abs(X) <= L / 2) & (np.abs(Y) <= L / 2)).astype(float)
    return {'x': x, 'y': y, 'X': X, 'Y': Y, 'dx': dx, 'dy': dy, 'A': A, 'L': L}
