"""Validate the NCC scoring metric and mode ranking in Task A."""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from physics_core import make_grid
from forward_model import forward_model
from residual import residual
from infer_mode_shape import run_task_A


@pytest.fixture(scope='module')
def grid():
    return make_grid(L=5.0, N=128)


@pytest.fixture(scope='module')
def base_params():
    return dict(kx1=2, ky1=1, kx2=2, ky2=1, beta=0.0,
                sigma_x=0.028, sigma_y=0.028, rotation_angle=0.0)


def _wrong_params(base):
    return {**base, 'kx1': 1, 'ky1': 3, 'kx2': 1, 'ky2': 3}


def test_ncc_perfect_recovery(grid, base_params):
    """NCC == 1.0 when O_model is identical to O_obs."""
    O = forward_model(base_params, grid)
    ncc = residual(O, O, mode='ncc')
    assert abs(ncc - 1.0) < 1e-6, f"NCC should be 1.0, got {ncc}"


def test_ncc_in_range(grid, base_params):
    """NCC is in [-1, 1] for any pair of maps."""
    O_obs = forward_model(base_params, grid)
    O_wrong = forward_model(_wrong_params(base_params), grid)
    ncc = residual(O_wrong, O_obs, mode='ncc')
    assert -1.0 <= ncc <= 1.0, f"NCC out of range: {ncc}"


def test_ncc_correct_beats_wrong(grid, base_params):
    """NCC for the correct mode exceeds NCC for an incorrect mode."""
    O_obs = forward_model(base_params, grid)
    ncc_correct = residual(forward_model(base_params, grid), O_obs, mode='ncc')
    ncc_wrong = residual(forward_model(_wrong_params(base_params), grid), O_obs, mode='ncc')
    assert ncc_correct > ncc_wrong, (
        f"Correct NCC ({ncc_correct:.4f}) should exceed wrong NCC ({ncc_wrong:.4f})")


def test_task_A_ranks_correct_mode_first(grid, base_params):
    """Task A should rank (kx1=2, ky1=1) as the top mode."""
    O_obs = forward_model(base_params, grid)
    results = run_task_A(O_obs, grid, sigma_x=0.028, sigma_y=0.028,
                         rotation_angle=0.0, kmax=5, top_n=5)
    assert results['top1']['kx1'] == 2, (
        f"Expected kx1=2, got {results['top1']['kx1']}")
    assert results['top1']['ky1'] == 1, (
        f"Expected ky1=1, got {results['top1']['ky1']}")


def test_ncc_scores_decrease_monotonically(grid, base_params):
    """NCC scores in the Task A ranked output should be non-increasing."""
    O_obs = forward_model(base_params, grid)
    results = run_task_A(O_obs, grid, sigma_x=0.028, sigma_y=0.028,
                         rotation_angle=0.0, kmax=5, top_n=5)
    scores = results['df']['ncc_score'].values
    assert all(scores[i] >= scores[i + 1] - 1e-10 for i in range(len(scores) - 1)), (
        f"NCC scores not monotonically non-increasing: {scores}")
