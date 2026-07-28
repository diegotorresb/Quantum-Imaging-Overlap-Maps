# Code Plan: `overlap_inverse.py`

## Philosophy

The forward model is the linear map

$$\mathcal{O}(x_0, y_0) = \iint \varphi(x,y;\,\theta) \cdot g_{mn}(x-x_0, y-y_0)\, dx\, dy$$

where $\varphi$ is the mechanical mode shape, $g_{mn} = u_{mn} \cdot u_{00}$ is the optical kernel, and $\theta$ collects all free parameters. Inversion is decomposed into three progressively harder tasks depending on what is assumed known.

---

## Module 0 — `physics_core.py`

Shared module, extracted from the Streamlit app with no UI dependency.

**Functions to extract:**

- `drum_mode_square(x, y, L, beta, kx1, ky1, kx2, ky2)`
- `HG_1D(x, n, sigma)`
- `HG_2D(x, y, n, m, sigma_x, sigma_y)`
- `rotated_HG_2D(X, Y, n, m, sigma_x, sigma_y, rotation_angle)`
- `optical_fields_product(X, Y, m, n, sigx, sigy, x0, y0, rotation_angle, rel_strength)`
- `overlap_map(m, n, sigx, sigy, phi, X, Y, dx, dy, rotation_angle, rel_strength)` — FFT-convolution path kept as default

Also add a grid helper:

```python
def make_grid(L, N, chip_margin=0.0):
    """
    Returns (x, y, X, Y, dx, dy, A) where A is the membrane aperture mask.
    chip_margin=0 restricts the grid to the membrane itself.
    """
```

The `@st.cache_data` decorators become `@functools.lru_cache` on hashable scalar args, or are dropped and managed externally. The aperture mask `A` is computed once here and passed through — it is critical for correct mode normalization in the overlap integrals.

---

## Module 1 — `forward_model.py`

Single entry point called by all three task modules.

```python
def forward_model(params: dict, grid: dict, mn: tuple = (0, 0)) -> np.ndarray:
    """
    params keys:
        mechanical   — beta, kx1, ky1, kx2, ky2
        illumination — sigma_x, sigma_y, rotation_angle
    grid keys:
        X, Y, dx, dy, A, L
    mn:
        HG indices of the signal mode (default u_00 illumination)

    Returns O_model on the same grid as O_obs.
    """
    phi = drum_mode_square(grid['X'], grid['Y'], grid['L'],
                           params['beta'], params['kx1'], params['ky1'],
                           params['kx2'], params['ky2']) * grid['A']
    m, n = mn
    return overlap_map(m, n, params['sigma_x'], params['sigma_y'],
                       phi, grid['X'], grid['Y'], grid['dx'], grid['dy'],
                       params['rotation_angle'])
```

---

## Module 2 — `residual.py`

```python
def residual(O_model: np.ndarray, O_obs: np.ndarray,
             mode: str = 'ncc', normalize: bool = True) -> float:
    """
    Supported modes:
        'ncc'  — normalized cross-correlation (default; amplitude-invariant)
        'mse'  — mean squared error
        'l1'   — L1 norm (more robust to outlier pixels)

    normalize: if True, divides both maps by their L2 norm before comparison.
               Recommended since overall amplitude (L, beam power) is not informative.

    NCC is returned as a similarity (higher = better), MSE and L1 as costs (lower = better).
    """
```

**NCC formula:**

$$\mathrm{NCC} = \frac{\hat{O}_{obs} \cdot \hat{O}_{model}}{\|\hat{O}_{obs}\| \cdot \|\hat{O}_{model}\|}$$

where $\hat{O} = O / \|O\|_2$. NCC is the recommended default because the absolute amplitude of the overlap map depends on normalization conventions and beam power, which carry no shape information.

---

## Task A — `infer_mode_shape.py`

**Assumed known:** `sigma_x`, `sigma_y`, `rotation_angle`, `L`, grid `N`.

**Estimated:** `(kx1, ky1)`, and optionally `(kx2, ky2, beta)`.

### Step 1 — Single-mode scan

For each `(kx, ky)` in `range(1, kmax+1)²`:

```python
O1 = forward_model({'beta': 0, 'kx1': kx, 'ky1': ky, 'kx2': kx, 'ky2': ky,
                    'sigma_x': sx, 'sigma_y': sy, 'rotation_angle': theta}, grid)
score = ncc(O1, O_obs)
```

Returns a ranked DataFrame of all candidates. At `kmax = 6` this is 36 evaluations — cheap even without caching.

### Step 2 — Two-mode linear solve

For each pair `((kx1,ky1), (kx2,ky2))` from the top-N shortlist of Step 1, the model is **linear in `beta`**:

$$\mathcal{O}_{obs} \approx \mathcal{O}_1 + \beta\, \mathcal{O}_2$$

Flattening to vectors and solving:

```python
A_mat = np.column_stack([O1.ravel(), O2.ravel()])   # shape (N², 2)
coeffs, _, _, _ = np.linalg.lstsq(A_mat, O_obs.ravel(), rcond=None)
# coeffs = [scale_1, scale_2]; recover beta = scale_2 / scale_1
```

This gives the best-fit `beta` for each candidate pair analytically, so the discrete search over the shortlist is $O(N^2)$ per pair with no iterative solver.

### Output

Ranked DataFrame with columns `[kx1, ky1, kx2, ky2, beta, ncc_score, rmse]`, plus the best-fit `O_model` array and the inferred mode map $\varphi_\mathrm{fit}$.

---

## Task B — `infer_illumination.py`

**Assumed known:** `beta`, `kx1`, `ky1`, `kx2`, `ky2`, `L`.

**Estimated:** `sigma_x`, `sigma_y`, `rotation_angle` (continuous, nonlinear).

### Step 1 — Initialization from moments

Estimate a starting `sigma` from the second-moment width of `O_obs`. The overlap map of a Gaussian kernel convolved with a slowly varying mode has a spatial width $\approx \sqrt{\sigma_\mathrm{mode}^2 + \sigma_\mathrm{beam}^2}$, usable as an upper bound for the initial guess.

### Step 2 — Global + local optimization

```python
from scipy.optimize import differential_evolution, minimize

def objective(p):
    sx, sy, theta = p
    O_model = forward_model({..., 'sigma_x': sx, 'sigma_y': sy,
                             'rotation_angle': theta}, grid)
    return -ncc(O_model, O_obs)   # maximize NCC

bounds = [(sigma_min, sigma_max), (sigma_min, sigma_max), (-90.0, 90.0)]

# Global search (handles rotation degeneracy)
result_global = differential_evolution(objective, bounds,
                                       maxiter=300, tol=1e-5, seed=42,
                                       workers=-1)   # parallel evaluations
# Local refinement from best point
result_local = minimize(objective, result_global.x,
                        method='Nelder-Mead',
                        options={'xatol': 1e-5, 'fatol': 1e-6})
```

Differential evolution is used first (global, handles the rotation degeneracy at `theta = 0` when `sigma_x ≈ sigma_y`), followed by Nelder-Mead refinement. For isotropic beams, fix `sigma_x = sigma_y` to reduce the search to 2D and avoid the degeneracy entirely.

### Step 3 — Parameter uncertainties

Compute the Jacobian of `O_model` w.r.t. `(sigma_x, sigma_y, theta)` at the optimum via `scipy.optimize.approx_fprime`, then propagate to parameter covariance:

$$\mathrm{Cov}(\hat{\theta}) \approx \sigma_r^2 \left(J^T J\right)^{-1}$$

where $\sigma_r^2$ is the residual variance per pixel. Report $1\sigma$ intervals on all three beam parameters.

### Output

`(sigma_x_fit, sigma_y_fit, theta_fit)` with $1\sigma$ uncertainties, NCC score, RMSE, and `O_model`.

---

## Task C — `infer_joint.py` (optional)

Alternating minimization for the fully blind case where neither mode shape nor beam parameters are known.

**Algorithm:**

1. Initialize with a coarse Task A scan assuming a nominal beam waist (e.g., `sigma = L/20`).
2. Fix mode from Task A result → run Task B.
3. Fix beam from Task B result → re-run Task A (linear step, fast).
4. Repeat until $\Delta \mathrm{NCC} < \epsilon$ or `max_iter` reached.

This handles the coupling between mode shape estimation error and beam estimation bias. Convergence is typically fast (2–4 outer iterations) because the subproblems are individually well-conditioned.

```python
def infer_joint(O_obs, grid, kmax=5, max_outer=10, tol=1e-4):
    sigma_init = {'sigma_x': grid['L']/20, 'sigma_y': grid['L']/20, 'rotation_angle': 0.0}
    mode_params = run_task_A(O_obs, grid, **sigma_init, kmax=kmax, top_n=1)
    for _ in range(max_outer):
        beam_params = run_task_B(O_obs, grid, **mode_params)
        mode_params_new = run_task_A(O_obs, grid, **beam_params, kmax=kmax, top_n=1)
        if converged(mode_params_new, mode_params, beam_params, tol):
            break
        mode_params = mode_params_new
    return mode_params, beam_params
```

---

## `diagnostics.py`

```python
def diagnostics(O_obs, O_model, params, grid, outdir=None):
    """
    1. Residual map: (O_obs - O_model) / max(|O_obs|)  — normalized, dimensionless
    2. 1D cuts: horizontal, vertical, diagonal through centroid of O_obs
    3. NCC landscape: 2D scan over (sigma_x, sigma_y) with theta fixed — for Task B
    4. Parameter summary table (via tabulate or rich)
    5. Matplotlib 3-panel figure: O_obs | O_model | residual
    6. Optional: save figure and results dict to outdir
    """
```

The residual map is the primary diagnostic. Large-amplitude structure in the residual indicates either a wrong `(kx, ky)` assignment or an incorrect beam rotation. A smooth, near-zero residual (< 5% of peak) indicates a good fit. The NCC landscape plot is useful for Task B to verify that the optimizer found a global rather than local minimum.

---

## `overlap_inverse.py` — CLI entry point

```python
"""
Usage examples:

  # Task A: infer mechanical mode, beam parameters known
  python overlap_inverse.py --task A --input O_obs.npz \
      --sigma_x 0.028 --sigma_y 0.028 --rotation 0.0 --kmax 5

  # Task B: infer beam parameters, mechanical mode known
  python overlap_inverse.py --task B --input O_obs.npz \
      --kx1 2 --ky1 1 --beta 0.0

  # Task C: joint inference (fully blind)
  python overlap_inverse.py --task C --input O_obs.npz --kmax 5
"""

import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Overlap map inverse solver")

    parser.add_argument('--task', choices=['A', 'B', 'C'], required=True)
    parser.add_argument('--input', type=Path, required=True,
                        help="NPZ file with keys: O_obs, x, y, L")

    # Task A arguments (beam parameters known)
    parser.add_argument('--sigma_x',  type=float, help="Beam sigma_x [mm]")
    parser.add_argument('--sigma_y',  type=float, help="Beam sigma_y [mm]")
    parser.add_argument('--rotation', type=float, default=0.0, help="Rotation angle [deg]")
    parser.add_argument('--kmax',     type=int,   default=5,   help="Max mode index to scan")
    parser.add_argument('--top_n',    type=int,   default=5,   help="Top-N candidates for two-mode solve")

    # Task B arguments (mechanical mode known)
    parser.add_argument('--kx1',  type=int,   default=1)
    parser.add_argument('--ky1',  type=int,   default=1)
    parser.add_argument('--kx2',  type=int,   default=1)
    parser.add_argument('--ky2',  type=int,   default=1)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--isotropic', action='store_true',
                        help="Enforce sigma_x = sigma_y (reduces Task B to 2D search)")

    # Shared
    parser.add_argument('--loss',    choices=['ncc', 'mse', 'l1'], default='ncc')
    parser.add_argument('--outdir',  type=Path, default=Path('.'))
    parser.add_argument('--no-plot', action='store_true')

    args = parser.parse_args()

    # Load O_obs
    data = np.load(args.input)
    O_obs = data['O_obs']
    x, y  = data['x'], data['y']
    L     = float(data['L'])

    grid = make_grid_from_arrays(x, y, L)

    if args.task == 'A':
        results = run_task_A(O_obs, grid, args.sigma_x, args.sigma_y,
                             args.rotation, args.kmax, args.top_n, args.loss)
    elif args.task == 'B':
        results = run_task_B(O_obs, grid, args.kx1, args.ky1, args.kx2, args.ky2,
                             args.beta, args.isotropic, args.loss)
    elif args.task == 'C':
        results = infer_joint(O_obs, grid, args.kmax)

    diagnostics(O_obs, results['O_model'], results['params'], grid,
                outdir=args.outdir, no_plot=args.no_plot)

    # Save results
    out_path = args.outdir / f"results_task{args.task}.json"
    save_results(results, out_path)
    print(f"Results saved to {out_path}")

if __name__ == '__main__':
    main()
```

**Input NPZ format** (produced by the Streamlit export or generated synthetically):

| Key | Shape | Description |
|---|---|---|
| `O_obs` | `(N, N)` | Observed overlap map |
| `x` | `(N,)` | x coordinate vector [mm] |
| `y` | `(N,)` | y coordinate vector [mm] |
| `L` | scalar | Membrane side length [mm] |

---

## Testing strategy

All tests live in `tests/` and use `pytest`.

### `test_roundtrip.py`

Generate a synthetic `O_obs` from known parameters via `forward_model`, run the inverse, assert recovered parameters within tolerance. Run for all three tasks. This is the primary correctness check.

```python
@pytest.mark.parametrize("kx1,ky1,beta", [(2,1,0.0), (1,2,0.3), (3,1,-0.5)])
def test_task_A_roundtrip(kx1, ky1, beta):
    params_true = dict(kx1=kx1, ky1=ky1, kx2=1, ky2=2, beta=beta,
                       sigma_x=0.028, sigma_y=0.028, rotation_angle=0.0)
    grid = make_grid(L=5.0, N=256)
    O_obs = forward_model(params_true, grid)
    results = run_task_A(O_obs, grid, sigma_x=0.028, sigma_y=0.028, rotation=0.0)
    assert results['top1']['kx1'] == kx1
    assert results['top1']['ky1'] == ky1
    assert abs(results['top1']['beta'] - beta) < 0.05
```

### `test_linearity.py`

Verify that $\mathcal{O}(\varphi_1 + \beta\varphi_2) = \mathcal{O}(\varphi_1) + \beta\,\mathcal{O}(\varphi_2)$ numerically, confirming the linear beta-solve in Task A is valid and that no nonlinear normalization has been accidentally introduced.

### `test_residual_modes.py`

Confirm NCC = 1.0 for perfect recovery, and that NCC decreases monotonically as wrong `(kx, ky)` indices are substituted. This validates the scoring metric and the mode ordering of the ranked output.

---

## Key design decisions

**NCC over MSE as default loss.** The absolute amplitude of the overlap map depends on normalization, beam power, and membrane size — none of which carry shape information. NCC is invariant to overall scale and offset, making it the natural shape-fitting metric. MSE is provided as an alternative for cases where amplitude calibration is available.

**Discrete + linear for Task A, continuous nonlinear for Task B.** The mode shape parameters `(kx, ky)` are integers, so brute-force enumeration is exact and cheap — at `kmax = 6`, only 36 single-mode candidates. `beta` is recovered analytically via least squares given the integer pair. This avoids mixed-integer optimization entirely.

**Global search first in Task B.** The NCC landscape over `(sigma_x, sigma_y, theta)` can have local maxima, especially near the isotropic point where `sigma_x ≈ sigma_y` and `theta` is degenerate. Differential evolution with parallel workers resolves this robustly before the local refinement step.

**Rotation degeneracy handling.** For a $u_{00}$ (isotropic Gaussian) illumination with `sigma_x = sigma_y`, `rotation_angle` is globally degenerate and the optimizer will return a spurious value. Task B should detect this case (via the `--isotropic` flag or a post-hoc check on `|sigma_x - sigma_y| / sigma_x`) and either fix `theta = 0` or report the degeneracy explicitly.

**Separation of `physics_core.py`.** The Streamlit app and the inverse solver share the same physical model exactly. Keeping it in a single importable module guarantees consistency and makes unit-testing the physics independently of either UI straightforward.
