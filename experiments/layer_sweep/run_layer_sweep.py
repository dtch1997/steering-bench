""" Script to compute steerability over all layers and plot the results """

import torch
import numpy as np
import pathlib

from steering_vectors import train_steering_vector
from steering_bench.build_training_data import build_steering_vector_training_data
from steering_bench.core.evaluate import evaluate_propensities_on_dataset
from steering_bench.utils.torch import load_model_with_quantization, EmptyTorchCUDACache
from steering_bench.dataset import build_dataset, DatasetSpec
from steering_bench.core.format import LlamaChatFormatter
from steering_bench.core.pipeline import Pipeline
from steering_bench.core.propensity import LogProbDifference
from steering_bench.core.hook import SteeringHook
from steering_bench.metric import get_steerability_slope

multipliers = np.array([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])

curr_dir = pathlib.Path(__file__).parent.absolute()


if __name__ == "__main__":

    save_dir = curr_dir / "layer_sweep_results"

    dataset_name = "corrigible-neutral-HHH"
    train_spec = DatasetSpec(name=dataset_name, split="0%:10%", seed=0)
    test_spec = DatasetSpec(name=dataset_name, split="99%:100%", seed=0)
    train_dataset = build_dataset(train_spec)
    test_dataset = build_dataset(test_spec)

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model, tokenizer = load_model_with_quantization(model_name, load_in_8bit=True)
    formatter = LlamaChatFormatter()
    pipeline = Pipeline(model=model, tokenizer=tokenizer, formatter=formatter)

    # Train the steering vector, or load a saved one
    sv_save_path = save_dir / f"steering_vector.pt"
    if sv_save_path.exists():
        steering_vector = torch.load(sv_save_path)
    else:
        training_data = build_steering_vector_training_data(pipeline, train_dataset)
        steering_vector = train_steering_vector(
            pipeline.model,
            pipeline.tokenizer,
            training_data,
        )
        torch.save(steering_vector, sv_save_path)

    # Evaluate propensity and steerability
    propensity_score = LogProbDifference()
    steerabilities: dict[int, float] = {}
    for layer in range(32):
        steering_hook = SteeringHook(
            steering_vector,
            direction_multiplier=0.0,  # Will be overwritten by evaluate_propensities
            layer=layer,
            patch_generation_tokens_only=True,
            skip_first_n_generation_tokens=1,
            patch_operator="add",
        )

        with EmptyTorchCUDACache():
            print(f"Running layer {layer}")
            propensities = evaluate_propensities_on_dataset(
                pipeline,
                steering_hook,
                test_dataset,
                propensity_fn=propensity_score,
                multipliers=multipliers,
            )

            steerability = get_steerability_slope(multipliers, propensities)
            print(
                f"Steerability slope: {steerability.mean():.2f} +- {steerability.std():.2f}"
            )
            steerabilities[layer] = steerability.mean()

        # Save multipliers, propensities, steerabilities
        np.save(save_dir / f"multipliers.npy", multipliers)
        np.save(save_dir / f"propensities_layer_{layer}.npy", propensities)
