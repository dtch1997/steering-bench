from jaxtyping import Float
import numpy as np

from scipy.stats import spearmanr


def get_steerability_slope(
    multipliers: Float[np.ndarray, "n_multipliers"],  # noqa
    propensities: Float[np.ndarray, "batch n_multipliers"],  # noqa
) -> Float[np.ndarray, "batch"]:  # noqa
    # NOTE: Assumes multipliers are the same for all datasets
    slope, _ = np.polyfit(multipliers, propensities.T, 1)
    return slope


def get_steerability_residuals(
    multipliers: Float[np.ndarray, "n_multipliers"],  # noqa
    propensities: Float[np.ndarray, "batch n_multipliers"],  # noqa
) -> Float[np.ndarray, "batch"]:  # noqa
    # NOTE: Assumes multipliers are the same for all datasets
    (_, _), residuals, _, _, _ = np.polyfit(multipliers, propensities.T, 1, full=True)
    return residuals


def get_steerabilty_spearman(
    multipliers: Float[np.ndarray, "n_multipliers"],  # noqa
    propensities: Float[np.ndarray, "batch n_multipliers"],  # noqa
) -> Float[np.ndarray, "batch"]:  # noqa
    """Compute the Spearman correlation between multipliers and propensities"""
    # batch_size = propensities.shape[0]
    # multipliers = multipliers[np.newaxis, :].repeat(batch_size, axis=0)
    result = spearmanr(multipliers, propensities, axis=1)
    return result.statistic[0, 1:]  # type: ignore
