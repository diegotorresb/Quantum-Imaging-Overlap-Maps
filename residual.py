import numpy as np


def residual(O_model: np.ndarray, O_obs: np.ndarray,
             mode: str = 'ncc', normalize: bool = True) -> float:
    """
    mode: 'ncc' (similarity, higher=better), 'mse' or 'l1' (cost, lower=better)
    normalize: divide both maps by their L2 norm before comparison
    NCC is amplitude-invariant and the recommended default.
    """
    O_m = O_model.ravel().astype(float)
    O_o = O_obs.ravel().astype(float)

    if normalize:
        norm_m = np.linalg.norm(O_m)
        norm_o = np.linalg.norm(O_o)
        O_m = O_m / norm_m if norm_m > 0 else O_m
        O_o = O_o / norm_o if norm_o > 0 else O_o

    if mode == 'ncc':
        return float(np.dot(O_o, O_m))
    elif mode == 'mse':
        return float(np.mean((O_m - O_o) ** 2))
    elif mode == 'l1':
        return float(np.mean(np.abs(O_m - O_o)))
    else:
        raise ValueError(f"Unknown residual mode '{mode}'. Choose 'ncc', 'mse', or 'l1'.")
