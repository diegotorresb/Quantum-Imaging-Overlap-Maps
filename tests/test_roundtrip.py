"""Roundtrip tests: generate synthetic O_obs from known params, invert, check recovery."""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from physics_core import make_grid
from forward_model import forward_model
from infer_mode_shape import run_task_A
from infer_illumination import run_task_B


@pytest.fixture(scope='module')
def grid():
    return make_grid(L=5.0, N=128)


# kx2=1, ky2=3 chosen to avoid degenerate case where secondary == primary
@pytest.mark.parametrize("kx1,ky1,beta", [(2, 1, 0.0), (1, 2, 0.3), (3, 1, -0.5)])
def test_task_A_roundtrip(grid, kx1, ky1, beta):
    params_true = dict(kx1=kx1, ky1=ky1, kx2=1, ky2=3, beta=beta,
                       sigma_x=0.028, sigma_y=0.028, rotation_angle=0.0)
    O_obs = forward_model(params_true, grid)
    results = run_task_A(O_obs, grid, sigma_x=0.028, sigma_y=0.028,
                         rotation_angle=0.0, kmax=5, top_n=5)
    assert results['top1']['kx1'] == kx1, (
        f"Expected kx1={kx1}, got {results['top1']['kx1']}")
    assert results['top1']['ky1'] == ky1, (
        f"Expected ky1={ky1}, got {results['top1']['ky1']}")
    assert abs(results['top1']['beta'] - beta) < 0.05, (
        f"Expected beta≈{beta}, got {results['top1']['beta']:.4f}")


@pytest.mark.slow
def test_task_B_roundtrip(grid):
    """Task B roundtrip — slow due to differential evolution."""
    params_true = dict(kx1=2, ky1=1, kx2=2, ky2=1, beta=0.0,
                       sigma_x=0.040, sigma_y=0.035, rotation_angle=15.0)
    O_obs = forward_model(params_true, grid)
    results = run_task_B(O_obs, grid, kx1=2, ky1=1, kx2=2, ky2=1,
                         beta=0.0, isotropic=False)
    assert abs(results['sigma_x'] - 0.040) < 0.005, (
        f"sigma_x: expected 0.040, got {results['sigma_x']:.4f}")
    assert abs(results['sigma_y'] - 0.035) < 0.005, (
        f"sigma_y: expected 0.035, got {results['sigma_y']:.4f}")
    assert abs(results['rotation_angle'] - 15.0) < 2.0, (
        f"rotation: expected 15.0, got {results['rotation_angle']:.2f}")
