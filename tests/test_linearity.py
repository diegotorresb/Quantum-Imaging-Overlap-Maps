"""Verify that the overlap integral is linear in the mechanical mode shape."""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from physics_core import make_grid, drum_mode_square, overlap_map


@pytest.fixture(scope='module')
def grid():
    return make_grid(L=5.0, N=64)


def test_overlap_linearity(grid):
    """O(phi1 + beta*phi2) == O(phi1) + beta*O(phi2) to floating-point precision."""
    X, Y, dx, dy, A, L = (grid['X'], grid['Y'], grid['dx'],
                           grid['dy'], grid['A'], grid['L'])
    sx, sy = 0.028, 0.028

    phi1 = drum_mode_square(X, Y, L, 0.0, 2, 1, 2, 1) * A
    phi2 = drum_mode_square(X, Y, L, 0.0, 1, 2, 1, 2) * A

    for beta in [0.3, -0.5, 1.0]:
        phi_combined = phi1 + beta * phi2
        O1 = overlap_map(0, 1, sx, sy, phi1, X, Y, dx, dy)
        O2 = overlap_map(0, 1, sx, sy, phi2, X, Y, dx, dy)
        O_combined = overlap_map(0, 1, sx, sy, phi_combined, X, Y, dx, dy)
        np.testing.assert_allclose(
            O_combined, O1 + beta * O2, rtol=1e-5, atol=1e-14,
            err_msg=f"Linearity failed for beta={beta}"
        )


def test_overlap_linearity_u00(grid):
    """Linearity holds for u_00 kernel (m=n=0) as well."""
    X, Y, dx, dy, A, L = (grid['X'], grid['Y'], grid['dx'],
                           grid['dy'], grid['A'], grid['L'])
    sx, sy = 0.028, 0.028

    phi1 = drum_mode_square(X, Y, L, 0.0, 1, 1, 1, 1) * A
    phi2 = drum_mode_square(X, Y, L, 0.0, 2, 2, 2, 2) * A
    beta = 0.5

    O1 = overlap_map(0, 0, sx, sy, phi1, X, Y, dx, dy)
    O2 = overlap_map(0, 0, sx, sy, phi2, X, Y, dx, dy)
    O_both = overlap_map(0, 0, sx, sy, phi1 + beta * phi2, X, Y, dx, dy)
    np.testing.assert_allclose(O_both, O1 + beta * O2, rtol=1e-5, atol=1e-14)
