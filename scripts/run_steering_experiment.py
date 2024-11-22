import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector, guess_and_enhance_layer_config
from steering_bench.utils.build_training_data import build_steering_vector_training_data
from steering_bench.utils.torch import load_model_with_quantization

from steering_bench.dataset import build_dataset, DatasetSpec
from steering_bench.format import IdentityFormatter
from steering_bench.pipeline import Pipeline
from steering_bench.hook import SteeringHook
from steering_bench.evaluate import evaluate, LogProbDifference, NormalizedPositiveProbability
from steering_bench.metric import get_steerability_slope

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    dataset_name = "power-seeking-inclination"
    train_spec = DatasetSpec(name=dataset_name, split = "0%:1%", seed = 0)
    test_spec = DatasetSpec(name=dataset_name, split = "99%:100%", seed = 0)
    train_dataset = build_dataset(train_spec)
    test_dataset = build_dataset(test_spec)

    model, tokenizer = load_model_with_quantization(model_name)

    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    formatter = IdentityFormatter()
    pipeline = Pipeline(model=model, tokenizer=tokenizer, formatter=formatter)

    training_data = build_steering_vector_training_data(pipeline, train_dataset)
    steering_vector = train_steering_vector(
        pipeline.model, 
        pipeline.tokenizer,
        training_data,
    )

    # Now evaluate the SV
    evaluators= [
        LogProbDifference(),
        NormalizedPositiveProbability(),
    ]

    multipliers = np.array([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
    propensities = np.zeros((len(test_dataset), len(multipliers)))

    for multiplier_idx, multiplier in enumerate(multipliers):
        steering_hook = SteeringHook(
            steering_vector,
            direction_multiplier=multiplier,
            layer = 8,
            layer_config = guess_and_enhance_layer_config(pipeline.model),
        )
        pipeline.hooks.append(steering_hook)
        result = evaluate(pipeline, test_dataset, evaluators)
        for test_idx, pred in enumerate(result.predictions):
            propensities[test_idx, multiplier_idx] = pred.metrics["logprob_diff"]

    # calculate steerability 
    slope = get_steerability_slope(multipliers, propensities)

    import seaborn as sns 
    import matplotlib.pyplot as plt
    sns.set_theme()

    # Propensity curve
    plt.figure()
    plt.plot(multipliers, propensities.T[:, :5])
    plt.xlabel("Multiplier")
    plt.ylabel("Logprob Difference")
    plt.title("Propensity Curve")

    # Histplot of the slope 
    plt.figure()
    sns.histplot(slope)

