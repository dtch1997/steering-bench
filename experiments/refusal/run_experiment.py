"""Script to do steering with the refusal direction"""

import torch
import numpy as np
import pathlib

from steering_vectors import train_steering_vector
from steering_bench.build_training_data import build_steering_vector_training_data
from steering_bench.core.evaluate import evaluate
from steering_bench.utils.torch import load_model_with_quantization
from steering_bench.dataset import build_dataset, DatasetSpec
from steering_bench.core.format import Formatter
from steering_bench.core.pipeline import Pipeline
from steering_bench.core.propensity import LogProbDifference
from steering_bench.core.hook import SteeringHook

curr_dir = pathlib.Path(__file__).parent.absolute()


if __name__ == "__main__":
    save_dir = curr_dir / "results"
    save_dir.mkdir(exist_ok=True)

    # Load the dataset
    train_spec = DatasetSpec(name="refusal_train", split="0%:10%", seed=0)
    train_dataset = build_dataset(train_spec)

    # Load the model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model, tokenizer = load_model_with_quantization(model_name, load_in_8bit=True)
    formatter = Formatter()
    pipeline = Pipeline(model=model, tokenizer=tokenizer, formatter=formatter)

    # Train the steering vector, or load a saved one
    sv_save_path = save_dir / "steering_vector.pt"
    if sv_save_path.exists():
        print("Loading steering vector")
        steering_vector = torch.load(sv_save_path)
    else:
        print("Training steering vector")
        training_data = build_steering_vector_training_data(pipeline, train_dataset)
        steering_vector = train_steering_vector(
            pipeline.model,
            pipeline.tokenizer,
            training_data,
        )
        torch.save(steering_vector, sv_save_path)

    # Evaluate propensity and steerability
    multipliers = np.array([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
    propensity_score = LogProbDifference()
    steerabilities: dict[int, float] = {}

    layer = 13
    propensity_save_path = save_dir / f"propensities_layer_{layer}.npy"

    # Create the steering hook, which applies the steering vector to the model
    steering_hook = SteeringHook(
        steering_vector,
        direction_multiplier=0.01,  # Placeholder value; will be overwritten by evaluate_propensities
        layer=layer,
        patch_generation_tokens_only=False,
        skip_first_n_generation_tokens=0,
        patch_operator="ablate_then_add",
    )

    # Evaluate on harmful test data
    test_spec = DatasetSpec(name="harmful_instructions_test", split="0%:10", seed=0)
    test_dataset = build_dataset(test_spec)

    # No steering
    prediction = evaluate(pipeline, test_dataset[0])
    print(prediction)

    # Ablating to very small refusal component
    steering_hook.direction_multiplier = 0.0001
    with pipeline.use_hooks([steering_hook]):
        prediction = evaluate(pipeline, test_dataset[0])
    print(prediction)

    # NOTE: This didn't give very crisp results
    # TODO: check if the ablate_then_add operator is working correctly
    # TODO: Try reproducing some figures from the paper
