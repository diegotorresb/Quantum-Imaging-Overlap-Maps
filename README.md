# Quantum Imaging Overlap Analysis

Tools for computing and inverting overlap maps between mechanical and optical modes in quantum optomechanics. The project has two components: an interactive Streamlit web app for forward simulation, and a command-line inverse solver for recovering mode or beam parameters from an observed overlap map.

---

## Project Structure

```
Quantum-Imaging-Overlap-Maps/
├── streamlit_app.py          # Interactive web app (forward model)
├── overlap_inverse.py        # CLI inverse solver entry point
├── physics_core.py           # Shared physics functions (no UI dependency)
├── forward_model.py          # Forward model used by all inverse tasks
├── residual.py               # NCC / MSE / L1 residual metrics
├── infer_mode_shape.py       # Task A: infer mechanical mode indices
├── infer_illumination.py     # Task B: infer beam parameters
├── infer_joint.py            # Task C: joint blind inference
├── diagnostics.py            # Residual maps and diagnostic figures
├── tests/
│   ├── test_linearity.py     # Verify overlap linearity in phi
│   ├── test_residual_modes.py# NCC metric and Task A ranking
│   └── test_roundtrip.py     # Synthetic roundtrip for Tasks A and B
├── requirements.txt
└── HG_overlap_v2.ipynb       # Original development notebook
```

---

## Installation

```bash
pip install -r requirements.txt
```

For the inverse solver only (no Streamlit required):

```bash
pip install numpy scipy pandas matplotlib
```

---

## Streamlit Web App

The web app computes the forward overlap map interactively. All parameters are controlled from the sidebar.

### Running

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in a browser.

### What it computes

The overlap map is the convolution of the mechanical mode shape with the optical kernel:

```
O(x0, y0) = integral phi(x,y) * g_mn(x - x0, y - y0) dx dy
```

where `phi` is the drum mode of a square membrane and `g_mn = u_mn * u_00` is the product of two Hermite-Gaussian modes.

### Controls

**Grid settings**
- Membrane side length `L` [mm]
- Background margin (opaque chip region around the membrane)
- Grid resolution `N` (points per side)

**Mechanical mode**
- `beta`: mixing coefficient between two drum modes
- `kx1, ky1`: wavenumber indices of the primary mode
- `kx2, ky2`: wavenumber indices of the secondary mode

**Optical mode**
- `m, n`: Hermite-Gaussian mode indices of the signal mode
- Beam parameters: sigma or optical waist (convertible in-app)
- Rotation angle [deg]
- Relative beam strength

### Tabs

**Matplotlib tab**
- Three-panel figure: mechanical mode, optical kernel, overlap map
- Plot value type: real, absolute, or magnitude squared
- Cut analysis: diagonal or custom point-to-point
- Rotated overlap map visualization
- CSV export of cut data and 2D overlap maps

**Plotly tab**
- 2D heatmaps and 3D surface plots of the same three quantities
- Interactive hover with position and overlap value

### Exported CSV formats

The Streamlit app can export data in two formats, both readable by the inverse solver:

| Format | Description | Columns |
|---|---|---|
| Flattened | One row per grid point | `x_mm`, `y_mm`, `overlap_real` |
| 2D array | Matrix form with coordinate labels | column headers `x_<value>`, row index `y_<value>` |

---

## Inverse Solver

The inverse solver infers unknown parameters from an observed overlap map `O_obs`. Three tasks are supported depending on what is known.

### Forward model

```
O_model = forward_model(params, grid)
```

`params` keys: `kx1, ky1, kx2, ky2, beta, sigma_x, sigma_y, rotation_angle`

### Task A: infer mechanical mode (beam known)

Recovers the drum mode wavenumber indices `(kx1, ky1)`, `(kx2, ky2)`, and mixing coefficient `beta`, given known beam parameters.

**Algorithm**
1. Brute-force single-mode scan over all `(kx, ky)` in `[1..kmax]^2`, ranked by |NCC|.
2. For each pair in the top-N shortlist, solve for `beta` analytically via least squares (linear in `beta`).

This avoids any iterative solver. At `kmax=5` the scan is 25 forward model evaluations.

**Usage**
```bash
python overlap_inverse.py --task A --input O_obs.npz \
    --sigma_x 0.028 --sigma_y 0.028 --rotation 0.0 --kmax 5 --top_n 5
```

### Task B: infer beam parameters (mode known)

Recovers `sigma_x`, `sigma_y`, and `rotation_angle` given known mode indices and `beta`.

**Algorithm**
1. Moment-based estimate of beam width as initialization.
2. Differential evolution (global search, handles rotation degeneracy).
3. Nelder-Mead refinement from the best global solution.
4. Jacobian-based 1-sigma parameter uncertainties via central differences.

**Usage**
```bash
python overlap_inverse.py --task B --input O_obs.npz \
    --kx1 2 --ky1 1 --kx2 2 --ky2 1 --beta 0.0
```

Add `--isotropic` to enforce `sigma_x = sigma_y` and reduce the search to 2D (avoids rotation degeneracy for round beams).

### Task C: joint blind inference

Alternates Task A and Task B until the NCC improvement per outer iteration falls below a threshold.

**Algorithm**
1. Run Task A with a nominal beam waist (`sigma = L/20`).
2. Fix mode from Task A, run Task B.
3. Fix beam from Task B, re-run Task A (fast linear step).
4. Repeat until convergence or `max_outer` iterations.

**Usage**
```bash
python overlap_inverse.py --task C --input O_obs.npz --kmax 5
```

### Input file formats

**NPZ (recommended)**

Keys: `O_obs` (N x N array), `x` (N,), `y` (N,), `L` (scalar, membrane side length in mm).

```python
import numpy as np
np.savez('O_obs.npz', O_obs=my_map, x=x_vec, y=y_vec, L=5.0)
```

**CSV (from Streamlit export)**

Both formats exported by the Streamlit app are accepted. Pass `--L` to specify the membrane size if it is not inferable from the grid.

```bash
# Flattened format (x_mm, y_mm, overlap_real columns)
python overlap_inverse.py --task A --input scan_flattened.csv --L 5.0 \
    --sigma_x 0.028 --sigma_y 0.028

# 2D array format (x_<val> column headers, y_<val> row index)
python overlap_inverse.py --task A --input scan_2d_array.csv --L 5.0 \
    --sigma_x 0.028 --sigma_y 0.028
```

### Full CLI reference

```
usage: overlap_inverse.py [-h] --task {A,B,C} --input INPUT [--L L]
                          [--sigma_x SIGMA_X] [--sigma_y SIGMA_Y]
                          [--rotation ROTATION] [--kmax KMAX] [--top_n TOP_N]
                          [--kx1 KX1] [--ky1 KY1] [--kx2 KX2] [--ky2 KY2]
                          [--beta BETA] [--isotropic]
                          [--loss {ncc,mse,l1}] [--outdir OUTDIR] [--no-plot]

  --task {A,B,C}       A=infer mode shape, B=infer beam, C=joint (blind)
  --input              Observation file (.npz or .csv)
  --L                  Membrane side length [mm] (required for CSV if not
                       inferable from grid extents)
  --sigma_x            Beam sigma_x [mm] (Task A)
  --sigma_y            Beam sigma_y [mm] (Task A)
  --rotation           Beam rotation angle [deg] (Task A, default 0)
  --kmax               Max mode index to scan (Task A/C, default 5)
  --top_n              Top-N candidates for two-mode solve (Task A, default 5)
  --kx1/ky1/kx2/ky2   Mode indices (Task B)
  --beta               Mode mixing coefficient (Task B)
  --isotropic          Enforce sigma_x = sigma_y (Task B)
  --loss               Loss function: ncc (default), mse, l1
  --outdir             Output directory (default: current directory)
  --no-plot            Suppress diagnostic figure
```

Outputs saved to `--outdir`:
- `results_taskA.json` / `results_taskB.json` / `results_taskC.json`
- `diagnostics.png` (O_obs | O_model | normalized residual)

---

## Running the Tests

```bash
# Fast tests only (linearity, NCC metric, Task A roundtrip)
pytest tests/ -k "not slow"

# All tests including Task B roundtrip (slow, uses differential evolution)
pytest tests/
```

---

## Physics Background

**Mechanical mode** (square membrane, Dirichlet boundary conditions):

```
phi(x,y) = sin(pi*kx1*(x + L/2)/L) * sin(pi*ky1*(y + L/2)/L)
           + beta * sin(pi*ky2*(y + L/2)/L) * sin(pi*kx2*(x + L/2)/L)
```

**Optical kernel** (product of Hermite-Gaussian modes):

```
g_mn(x,y) = u_mn(x,y) * u_00(x,y)
```

where `u_mn` is the (m,n) HG mode with beam widths `sigma_x`, `sigma_y`, and optional rotation angle.

**Overlap map** (computed via FFT convolution):

```
O(x0, y0) = sum_{x,y} phi(x,y) * g_mn(x - x0, y - y0) * dx * dy
```

**NCC loss** (default, scale-invariant):

```
NCC = O_obs_normalized . O_model_normalized
```

where each map is divided by its L2 norm before taking the dot product. NCC = 1 for a perfect match.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | >=1.24.0 | Array operations, FFT |
| scipy | >=1.10.0 | Special functions, optimization, convolution |
| matplotlib | >=3.7.0 | Static plots and diagnostic figures |
| pandas | >=2.0.0 | CSV I/O, results DataFrames |
| plotly | >=5.15.0 | Interactive plots (web app only) |
| streamlit | >=1.28.0 | Web app (not needed for CLI) |
| streamlit-vertical-slider | >=0.0.1 | Custom sliders (web app only) |
