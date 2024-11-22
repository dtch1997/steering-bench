import seaborn as sns
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from steering_bench.metric import get_steerability_slope

sns.set_theme()

curr_dir = pathlib.Path(__file__).parent.absolute()
save_dir = curr_dir / "layer_sweep_results"


def plot_propensity_curve():
    propensities: dict[int, np.ndarray] = {}
    for layer in range(32):
        p_layer = np.load(save_dir / f"propensities_layer_{layer}.npy")
        propensities[layer] = p_layer
    multipliers = np.load(save_dir / "multipliers.npy")

    plt.figure()
    plt.plot(multipliers, propensities[13].T[:, :10])
    plt.show()


def plot_layer_response_curve():
    """Make a plot of the layer response curve, with error bars"""

    propensities: dict[int, np.ndarray] = {}
    for layer in range(32):
        p_layer = np.load(save_dir / f"propensities_layer_{layer}.npy")
        propensities[layer] = p_layer

    multipliers = np.load(save_dir / "multipliers.npy")

    # mean, std are over the dataset
    steerability_means = []
    steerability_stds = []

    for layer in range(32):
        p_layer = propensities[layer]
        steerabilities = get_steerability_slope(multipliers, p_layer)
        mean = steerabilities.mean()
        std = steerabilities.std()
        steerability_means.append(mean)
        steerability_stds.append(std)

    plt.figure()
    plt.errorbar(
        range(32),
        steerability_means,
        yerr=steerability_stds,
        fmt="o",
        capsize=5,
    )
    plt.show()


if __name__ == "__main__":
    plot_layer_response_curve()
    plot_propensity_curve()
