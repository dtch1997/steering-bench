# pyright: strict, reportMissingTypeStubs=false


from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean, stdev, StatisticsError
from tqdm import tqdm
from typing import Iterable, Sequence

from steering_bench.pipeline import TextProbs
from steering_bench.core import Example
from steering_bench.pipeline import Pipeline

import numpy as np
import logging

# Set up module-level logger
logger = logging.getLogger(__name__)


@dataclass
class EvalPrediction:
    positive_output_prob: TextProbs | None
    negative_output_prob: TextProbs | None
    # Example-level metrics
    metrics: dict[str, float]


@dataclass
class EvalResult:
    predictions: list[EvalPrediction]
    # Dataset-level metrics; e.g. mean, stds of example-level metrics
    metrics: dict[str, float]


class Evaluator(ABC):
    requires_generation: bool = False
    requires_probs: bool = False

    @abstractmethod
    def get_metric_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def score_prediction(self, prediction: EvalPrediction) -> float:
        raise NotImplementedError

    def __call__(self, predictions: Sequence[EvalPrediction]) -> dict[str, float]:
        pred_results = [self.score_prediction(pred) for pred in predictions]
        # Compute mean, stdev of metric

        metric: dict[str, float] = {}
        metric[f"mean_{self.get_metric_name()}"] = mean(pred_results)

        # NOTE: In the case where pred_results only has one element,
        # stdev will raise a StatisticsError. We handle this gracefully.
        try:
            metric[f"std_{self.get_metric_name()}"] = stdev(pred_results)
        except StatisticsError as _:
            pass

        return metric


class Accuracy(Evaluator):

    requires_probs = True

    def get_metric_name(self) -> str:
        return "mcq_acc"

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction, 1 if correct, 0 otherwise."""
        assert prediction.positive_output_prob is not None
        assert prediction.negative_output_prob is not None
        positive_output_prob = prediction.positive_output_prob.sum_logprobs
        negative_output_prob = prediction.negative_output_prob.sum_logprobs
        return 1.0 if positive_output_prob > negative_output_prob else 0.0


class LogProbDifference(Evaluator):
    """
    Evaluator that scores multiple choice examples by computing the average difference
    in logprob between the correct and incorrect outputs.
    """

    requires_probs = True

    def get_metric_name(self) -> str:
        return "logprob_diff"

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction based on difference in sum of logits."""

        # calculate difference in logits
        assert prediction.positive_output_prob is not None
        assert prediction.negative_output_prob is not None
        # Recall: logprob(A) - logprob(B) = logit(A) - logit(B)
        positive_output_logprob = prediction.positive_output_prob.sum_logprobs
        negative_output_logprob = prediction.negative_output_prob.sum_logprobs
        return positive_output_logprob - negative_output_logprob


class NormalizedPositiveProbability(Evaluator):
    """
    Evaluator that scores multiple choice examples by computing the
    normalized probability of the positive output
    """

    requires_probs = True

    def get_metric_name(self) -> str:
        return "pos_prob"

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """
        Normalize the probabilities of positive and negative outputs relative to each other
        NOTE: This returns actual probabilities, not logprobs
        """

        # calculate normalized logprobs
        assert prediction.positive_output_prob is not None
        assert prediction.negative_output_prob is not None
        positive_output_logprob = prediction.positive_output_prob.sum_logprobs
        negative_output_logprob = prediction.negative_output_prob.sum_logprobs

        # normalize by max to avoid underflow?
        max_logprob = max(positive_output_logprob, negative_output_logprob)
        positive_output_logprob = positive_output_logprob - max_logprob
        negative_output_logprob = negative_output_logprob - max_logprob

        # Calculate normalized probability
        positive_output_prob = np.exp(positive_output_logprob)
        negative_output_prob = np.exp(negative_output_logprob)
        return positive_output_prob / (positive_output_prob + negative_output_prob)


def evaluate(
    pipeline: Pipeline,
    dataset: Iterable[Example],
    evaluators: Sequence[Evaluator],
    show_progress: bool = True,
    tqdm_desc: str = "Evaluating",
) -> EvalResult:
    
    """ Evaluate the pipeline on a dataset. """

    # evaluate
    predictions: list[EvalPrediction] = []

    # TODO: support batching
    for _, example in enumerate(
        tqdm(dataset, disable=not show_progress, desc=tqdm_desc)
    ):
        logger.debug(
            f"Example full prompt: \n {pipeline.build_full_prompt(example.positive)}"
        )
        positive_probs = pipeline.calculate_output_logprobs(example.positive)
        negative_probs = pipeline.calculate_output_logprobs(example.negative)

        pred = EvalPrediction(
            positive_output_prob=positive_probs,
            negative_output_prob=negative_probs,
            metrics={},
        )
        example_metrics = {}
        for evaluator in evaluators:
            example_metrics[evaluator.get_metric_name()] = (
                evaluator.score_prediction(pred)
            )
        pred.metrics = example_metrics

        predictions.append(pred)

    dataset_metrics: dict[str, float] = {}
    for evaluator in evaluators:
        dataset_metrics.update(evaluator(predictions))
    return EvalResult(predictions, dataset_metrics)