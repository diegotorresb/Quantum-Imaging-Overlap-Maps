"""
Overlap map inverse solver — CLI entry point.

Usage examples:

  # Task A: infer mechanical mode, beam parameters known
  python overlap_inverse.py --task A --input O_obs.npz \\
      --sigma_x 0.028 --sigma_y 0.028 --rotation 0.0 --kmax 5

  # Task B: infer beam parameters, mechanical mode known
  python overlap_inverse.py --task B --input O_obs.npz \\
      --kx1 2 --ky1 1 --beta 0.0

  # Task C: joint inference (fully blind)
  python overlap_inverse.py --task C --input O_obs.npz --kmax 5

  # CSV input — flattened format (columns: x_mm, y_mm, overlap_real)
  python overlap_inverse.py --task A --input scan.csv --L 5.0 \\
      --sigma_x 0.028 --sigma_y 0.028

  # CSV input — 2D array format (x_<val> column headers, y_<val> row index)
  python overlap_inverse.py --task A --input scan_2d_array.csv --L 5.0 \\
      --sigma_x 0.028 --sigma_y 0.028

Input file formats
------------------
NPZ (.npz): keys O_obs (N×N), x (N,), y (N,), L (scalar)

CSV (.csv) — two formats produced by the Streamlit export:
  Flattened:  columns 'x_mm', 'y_mm', 'overlap_real'
  2D array:   column headers 'x_<value>', row index 'y_<value>'
              (leading comma in header = index column present)

For CSV files, --L sets the membrane side length [mm]. If omitted,
L is inferred as 2 * max(|x|), which is correct when the grid spans [-L/2, L/2].
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from physics_core import make_grid_from_arrays
from infer_mode_shape import run_task_A
from infer_illumination import run_task_B
from infer_joint import infer_joint
from diagnostics import diagnostics


def load_observation(path: Path, L_override: float = None):
    """Load O_obs, x, y, L from a .npz or .csv file."""
    suffix = path.suffix.lower()

    if suffix == '.npz':
        data = np.load(path)
        return data['O_obs'], data['x'], data['y'], float(data['L'])

    elif suffix == '.csv':
        with open(path) as fh:
            first_line = fh.readline().strip()

        if 'x_mm' in first_line and 'y_mm' in first_line:
            # Flattened format: x_mm, y_mm, overlap_real
            df = pd.read_csv(path)
            x_flat = df['x_mm'].values
            y_flat = df['y_mm'].values
            O_flat = df['overlap_real'].values
            x = np.unique(x_flat)
            y = np.unique(y_flat)
            # Reshape: rows vary with y, cols vary with x (matches indexing='xy')
            O_obs = O_flat.reshape(len(y), len(x))
            L = L_override if L_override is not None else float(np.abs(x).max() * 2)
            return O_obs, x, y, L

        elif first_line.startswith(',') or first_line.split(',')[0].strip().startswith('x_'):
            # 2D array format: column headers x_<val>, optional row index y_<val>
            has_index = first_line.startswith(',')
            df = pd.read_csv(path, index_col=0 if has_index else None)
            x = np.array([float(c.split('_', 1)[1]) for c in df.columns])
            if has_index:
                y = np.array([float(str(r).split('_', 1)[1]) for r in df.index])
            else:
                # Infer y from grid spacing
                step = x[1] - x[0] if len(x) > 1 else 1.0
                y = np.arange(len(df)) * step - np.abs(x).max()
            O_obs = df.values.astype(float)
            L = L_override if L_override is not None else float(np.abs(x).max() * 2)
            return O_obs, x, y, L

        else:
            raise ValueError(
                f"CSV format in '{path}' not recognized.\n"
                "Supported formats:\n"
                "  Flattened:  columns 'x_mm', 'y_mm', 'overlap_real'\n"
                "  2D array:   column headers 'x_<value>', row index 'y_<value>'\n"
                "Both are produced by the Streamlit 'Export 2D Rotated Map' button.\n"
                "Pass --L to specify membrane size when using CSV."
            )

    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Use .npz or .csv.")


def save_results(results: dict, path: Path):
    import pandas as pd
    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            continue
        elif isinstance(v, pd.DataFrame):
            serializable[k] = v.to_dict(orient='records')
        else:
            serializable[k] = v
    with open(path, 'w') as fh:
        json.dump(serializable, fh, indent=2)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlap map inverse solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--task', choices=['A', 'B', 'C'], required=True,
                        help="A=infer mode shape, B=infer beam, C=joint (blind)")
    parser.add_argument('--input', type=Path, required=True,
                        help="Observation file (.npz or .csv)")
    parser.add_argument('--L', type=float, default=None,
                        help="Membrane side length [mm] (required for CSV if not inferable)")

    # Task A arguments
    parser.add_argument('--sigma_x',  type=float, help="Beam sigma_x [mm] (Task A)")
    parser.add_argument('--sigma_y',  type=float, help="Beam sigma_y [mm] (Task A)")
    parser.add_argument('--rotation', type=float, default=0.0,
                        help="Beam rotation angle [deg] (Task A, default 0)")
    parser.add_argument('--kmax',     type=int,   default=5,
                        help="Max mode index to scan (Task A/C, default 5)")
    parser.add_argument('--top_n',    type=int,   default=5,
                        help="Top-N candidates for two-mode solve (Task A, default 5)")

    # Task B arguments
    parser.add_argument('--kx1',  type=int,   default=1, help="Mode index kx1 (Task B)")
    parser.add_argument('--ky1',  type=int,   default=1, help="Mode index ky1 (Task B)")
    parser.add_argument('--kx2',  type=int,   default=1, help="Mode index kx2 (Task B)")
    parser.add_argument('--ky2',  type=int,   default=1, help="Mode index ky2 (Task B)")
    parser.add_argument('--beta', type=float, default=0.0,
                        help="Mode mixing coefficient beta (Task B)")
    parser.add_argument('--isotropic', action='store_true',
                        help="Enforce sigma_x = sigma_y in Task B (2D search)")

    # Shared
    parser.add_argument('--loss',    choices=['ncc', 'mse', 'l1'], default='ncc',
                        help="Loss function (default: ncc)")
    parser.add_argument('--outdir',  type=Path, default=Path('.'),
                        help="Output directory for results and figures (default: .)")
    parser.add_argument('--no-plot', action='store_true', dest='no_plot',
                        help="Suppress diagnostic plot")

    args = parser.parse_args()

    O_obs, x, y, L = load_observation(args.input, args.L)
    grid = make_grid_from_arrays(x, y, L)
    print(f"Loaded: shape={O_obs.shape}, L={L:.3f} mm, "
          f"x=[{x.min():.3f}, {x.max():.3f}] mm")

    if args.task == 'A':
        if args.sigma_x is None or args.sigma_y is None:
            parser.error("Task A requires --sigma_x and --sigma_y")
        results = run_task_A(
            O_obs, grid,
            sigma_x=args.sigma_x, sigma_y=args.sigma_y,
            rotation_angle=args.rotation,
            kmax=args.kmax, top_n=args.top_n, loss=args.loss,
        )
        print("\n=== Task A Results (top candidates) ===")
        print(results['df'].head(10).to_string(index=False))

    elif args.task == 'B':
        results = run_task_B(
            O_obs, grid,
            kx1=args.kx1, ky1=args.ky1,
            kx2=args.kx2, ky2=args.ky2,
            beta=args.beta, isotropic=args.isotropic, loss=args.loss,
        )
        print("\n=== Task B Results ===")
        for k, v in results.items():
            if not isinstance(v, np.ndarray):
                print(f"  {k}: {v}")

    elif args.task == 'C':
        results = infer_joint(O_obs, grid, kmax=args.kmax)
        print("\n=== Task C Results ===")
        print("  Mode params:", results['mode_params'])
        print("  Beam params:", results['beam_params'])
        print(f"  NCC score:  {results['ncc_score']:.4f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    diagnostics(O_obs, results['O_model'], results['params'], grid,
                outdir=args.outdir, no_plot=args.no_plot)

    save_results(results, args.outdir / f"results_task{args.task}.json")


if __name__ == '__main__':
    main()
