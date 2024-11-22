# pyright: reportMissingTypeStubs=false

import abc
import numpy as np
import tqdm as tqdm

from typing import Sequence
from jaxtyping import Float
from dataclasses import dataclass

from steering_bench.core.pipeline import TextProbs
from steering_bench.core.types import Example
from steering_bench.core.pipeline import Pipeline
from steering_bench.core.hook import SteeringHook


@dataclass
class EvalPrediction:
    positive_output_prob: TextProbs | None
    negative_output_prob: TextProbs | None


class PropensityScore(abc.ABC):
    requires_generation: bool = False
    requires_probs: bool = False

    @abc.abstractmethod
    def get_metric_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def score_prediction(self, prediction: EvalPrediction) -> float:
        raise NotImplementedError

    def __call__(self, prediction: EvalPrediction) -> float:
        return self.score_prediction(prediction)


def evaluate(
    pipeline: Pipeline,
    example: Example,
) -> EvalPrediction:
    """Evaluate the pipeline on a dataset."""
    positive_probs = pipeline.calculate_output_logprobs(example.positive)
    negative_probs = pipeline.calculate_output_logprobs(example.negative)
    pred = EvalPrediction(
        positive_output_prob=positive_probs,
        negative_output_prob=negative_probs,
    )
    return pred


def evaluate_propensities(
    pipeline: Pipeline,
    hook: SteeringHook,
    example: Example,
    propensity_fn: PropensityScore,
    multipliers: Float[np.ndarray, " n_multipliers"],
) -> Float[np.ndarray, " n_multipliers"]:
    """Evaluate the propensity on a single example."""

    propensities: list[float] = []

    for multiplier in multipliers:
        hook.direction_multiplier = multiplier
        with pipeline.use_hooks([hook]):
            pred = evaluate(pipeline, example)
            propensity = propensity_fn(pred)
            propensities.append(propensity)
        hook.direction_multiplier = 0.0

    return np.array(propensities)


def evaluate_propensities_on_dataset(
    pipeline: Pipeline,
    hook: SteeringHook,
    dataset: Sequence[Example],
    propensity_fn: PropensityScore,
    multipliers: Float[np.ndarray, " n_multipliers"],
    *,
    show_progress: bool = True,
    desc: str = "Evaluating propensities",
) -> Float[np.ndarray, "n_examples n_multipliers"]:
    """Evaluate the propensity of the pipeline with the given hook on a dataset."""

    propensities: list[np.ndarray] = []

    for example in tqdm.tqdm(dataset, desc=desc, disable=not show_progress):
        propensities.append(
            evaluate_propensities(
                pipeline,
                hook,
                example,
                propensity_fn,
                multipliers,
            )
        )

    return np.stack(propensities, axis=0)
