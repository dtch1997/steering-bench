from typing import Iterable, Sequence

from steering_bench.data.load import Dataset
from steering_bench.core.pipeline import Pipeline
from steering_bench.evaluate import (
    evaluate,
    get_default_evaluators,
    Evaluator,
    EvalResult,
)
from steering_bench.steering import SteeringHook, SteeringConfig
from steering_vectors import SteeringVector


def evaluate_steering_vector(
    steering_config: SteeringConfig,
    pipeline: Pipeline,
    steering_vector: SteeringVector,
    dataset: Dataset,
    evaluators: Sequence[Evaluator] = get_default_evaluators(),
    show_progress: bool = True,
) -> EvalResult:
    """Evaluate the steering vector on the given dataset using the given config and pipeline."""

    # Create steering hook and add it to pipeline
    steering_hook = SteeringHook(
        steering_vector=steering_vector,
        config=steering_config,
    )

    # Evaluate the pipeline
    pipeline.hooks.clear()
    pipeline.hooks.append(steering_hook)
    result = evaluate(
        pipeline,
        dataset,
        evaluators=evaluators,
        show_progress=show_progress,
    )
    pipeline.hooks.clear()

    return result


def evaluate_steering_vector_sweep(
    steering_configs: Iterable[SteeringConfig],
    pipeline: Pipeline,
    steering_vector: SteeringVector,
    dataset: Dataset,
    evaluators: Sequence[Evaluator] = get_default_evaluators(),
    show_progress: bool = True,
):
    """Syntactic sugar to evaluate on multiple configs sequentially"""
    results = []
    for config in steering_configs:
        result = evaluate_steering_vector(
            config,
            pipeline,
            steering_vector,
            dataset,
            evaluators,
            show_progress,
        )
        results.append(result)
    return results


def make_sweep_layers_and_multipliers(
    config: SteeringConfig,
    layers: list[int],
    multipliers: list[float],
) -> list[SteeringConfig]:
    """Create a list of SteeringConfigs to sweep over layers and multipliers"""
    return [
        SteeringConfig(
            layer=layer,
            multiplier=multiplier,
            patch_generation_tokens_only=config.patch_generation_tokens_only,
            skip_first_n_generation_tokens=config.skip_first_n_generation_tokens,
            layer_config=config.layer_config,
            patch_operator=config.patch_operator,
        )
        for layer in layers
        for multiplier in multipliers
    ]
