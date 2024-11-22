""" Script to compute steerability over all layers and plot the results """

import numpy as np

from steering_vectors import train_steering_vector, guess_and_enhance_layer_config, SteeringVector
from steering_bench.utils.build_training_data import build_steering_vector_training_data
from steering_bench.utils.torch import load_model_with_quantization, EmptyTorchCUDACache
from steering_bench.dataset import build_dataset, DatasetSpec
from steering_bench.format import LlamaChatFormatter
from steering_bench.pipeline import Pipeline
from steering_bench.hook import SteeringHook
from steering_bench.evaluate import evaluate, LogProbDifference, NormalizedPositiveProbability
from steering_bench.metric import get_steerability_slope

multipliers = np.array([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])

def evaluate_propensities_at_layer(
    pipeline: Pipeline,
    layer: int,
    steering_vector: SteeringVector,
):
    """ Run steering at many different layers and evaluate propensities """

    evaluators= [
        LogProbDifference(),
        NormalizedPositiveProbability(),
    ]

    propensities = np.zeros((len(test_dataset), len(multipliers)))

    # get the propensities for each multiplier
    for multiplier_idx, multiplier in enumerate(multipliers):
        steering_hook = SteeringHook(
            steering_vector,
            direction_multiplier=multiplier,
            layer = layer,
            layer_config = guess_and_enhance_layer_config(pipeline.model),
        )
        pipeline.hooks.append(steering_hook)
        result = evaluate(pipeline, test_dataset, evaluators)
        for test_idx, pred in enumerate(result.predictions):
            propensities[test_idx, multiplier_idx] = pred.metrics["logprob_diff"]

    return propensities


if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    dataset_name = "corrigible-neutral-HHH"
    train_spec = DatasetSpec(name=dataset_name, split = "0%:10%", seed = 0)
    test_spec = DatasetSpec(name=dataset_name, split = "99%:100%", seed = 0)
    train_dataset = build_dataset(train_spec)
    test_dataset = build_dataset(test_spec)

    model, tokenizer = load_model_with_quantization(model_name, load_in_8bit=True)
    formatter = LlamaChatFormatter()
    pipeline = Pipeline(model=model, tokenizer=tokenizer, formatter=formatter)

    training_data = build_steering_vector_training_data(pipeline, train_dataset)
    steering_vector = train_steering_vector(
        pipeline.model, 
        pipeline.tokenizer,
        training_data,
    )

    slopes: dict[int, float] = {}
    for layer in range(32):
        with EmptyTorchCUDACache():
            print(f"Running layer {layer}")
            propensities = evaluate_propensities_at_layer(pipeline, layer, steering_vector)
            slope = get_steerability_slope(multipliers, propensities)
            print(f"Steerability slope: {slope.mean():.2f} +- {slope.std():.2f}")
            slopes[layer] = slope.mean()

    # Plot the results
    import seaborn as sns 
    import matplotlib.pyplot as plt
    sns.set_theme()

    xs = list(slopes.keys())
    ys = list(slopes.values())
    plt.plot(xs, ys)