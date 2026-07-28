import numpy as np
import matplotlib.pyplot as plt

from residual import residual


def diagnostics(O_obs, O_model, params, grid, outdir=None, no_plot=False):
    """
    1. Residual map: (O_obs - O_model) / max(|O_obs|)  — normalized, dimensionless
    2. NCC score, MSE, residual RMS
    3. Parameter summary table
    4. 3-panel figure: O_obs | O_model | residual
    5. Optionally saves figure to outdir/diagnostics.png
    """
    peak = float(np.max(np.abs(O_obs))) or 1.0
    res_map = (O_obs - O_model) / peak

    ncc = residual(O_model, O_obs, mode='ncc')
    mse = residual(O_model, O_obs, mode='mse', normalize=False)
    residual_rms = float(np.sqrt(np.mean(res_map ** 2)))

    print("\n=== Parameter Summary ===")
    for k, v in params.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nNCC score:    {ncc:.4f}")
    print(f"MSE:          {mse:.4e}")
    print(f"Residual RMS: {residual_rms:.4f}  (fraction of peak |O_obs|)")

    if not no_plot:
        x, y = grid['x'], grid['y']
        extent = [x.min(), x.max(), y.min(), y.max()]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axs[0].imshow(O_obs, extent=extent, origin='lower', cmap='viridis')
        axs[0].set_title('O_obs (measured)')
        axs[0].set_xlabel('x [mm]')
        axs[0].set_ylabel('y [mm]')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(O_model, extent=extent, origin='lower', cmap='viridis')
        axs[1].set_title('O_model (fit)')
        axs[1].set_xlabel('x [mm]')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        clim = max(0.05, float(np.percentile(np.abs(res_map), 95)))
        im2 = axs[2].imshow(res_map, extent=extent, origin='lower', cmap='RdBu_r',
                             vmin=-clim, vmax=clim)
        axs[2].set_title(f'Residual  NCC={ncc:.3f}  RMS={residual_rms:.3f}')
        axs[2].set_xlabel('x [mm]')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if outdir is not None:
            fig_path = outdir / 'diagnostics.png'
            fig.savefig(fig_path, dpi=150)
            print(f"Figure saved to {fig_path}")
        plt.show()

    return {'ncc': ncc, 'mse': mse, 'residual_rms': residual_rms}
