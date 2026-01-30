import numpy as np
from typing import Literal, Callable

from src.math import _compute_eigenvalues


def select_number_of_factors(
        returns_matrix: np.ndarray,
        method: Literal['threshold', 'scree', 'bai_ng'],
        max_factors: int = 50,
) -> int:
    """
    Select number of factors using selected criteria.
    """
    n_assets, T = returns_matrix.shape
    eigenvalues = _compute_eigenvalues(returns_matrix, T)
    max_factors = min(max_factors, len(eigenvalues))

    dispatch: dict[method, Callable[[], int]] = {
        'threshold': lambda: _threshold_method(eigenvalues, max_factors, n_assets / T),
        'scree': lambda: _scree_method(eigenvalues, max_factors),
        'bai_ng': lambda: _bai_ng_method(eigenvalues, max_factors, n_assets, T)
    }
    return dispatch[method]()


def _threshold_method(
        eigenvalues: np.ndarray,
        max_factors: int,
        gamma: float
) -> int:
    """
    Select eigenvalues above 1 + sqrt(y).
    """
    threshold = 1 + np.sqrt(gamma)
    return max(
        1,
        int(np.sum(eigenvalues[:max_factors] >= threshold))
    )


def _scree_method(
        eigenvalues: np.ndarray,
        max_factors: int,
        eps: float = 1e-10
) -> int:
    """
    Maximum value in log-eigenvalues.
    """
    log_eigs = np.log(eigenvalues[:max_factors] + eps)
    gaps = log_eigs[:-1] - log_eigs[1:]
    return max(
        1,
        int(np.argmax(gaps) + 1)
    )


def _bai_ng_method(
        eigenvalues: np.ndarray,
        max_factors: int,
        n_assets: int,
        T: int,
) -> int:
    """
    Penalty-based method.
    """
    def criterion(k: int) -> float:
        residual_var = np.sum(eigenvalues[k:])
        penalty = k * (n_assets + T) / (n_assets * T) * np.log(n_assets * T / (n_assets + T))
        return residual_var + penalty

    return min(
        range(1, max_factors + 1),
        key=criterion
    )