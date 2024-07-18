"""Infrastructure to evaluate a pipeline on a dataset"""

import numpy as np

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from statistics import mean, stdev, StatisticsError
from tqdm import tqdm
from typing import Callable, Iterable, Sequence

from steering_bench.core.pipeline import TextProbs
from steering_bench.core.types import ContrastivePair
from steering_bench.core.pipeline import Pipeline

# eval_hooks allow us to do custom stuff to the pipeline only during evaluation
# NOTE(dtch1997): I find it clunky to have both PipelineHook and EvalHook.
# I think we should merge EvalHook into PipelineHook, with a flag to indicate whether it's for evaluation or not.

EvalHook = Callable[[Pipeline], AbstractContextManager[None]]


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


class MultipleChoiceAccuracyEvaluator(Evaluator):
    """
    Evaluator that scores multiple choice examples by computing the probability of
    the correct output and comparing it to the probability of the incorrect outputs.
    """

    requires_probs = True

    def get_metric_name(self) -> str:
        return "mcq_acc"

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction, 1 if correct, 0 otherwise."""

        # the output might be longer than the expected depending on how many tokens we generate
        # so just verify that the expected output is a prefix of the generated output
        assert prediction.positive_output_prob is not None
        assert prediction.negative_output_prob is not None
        positive_output_prob = prediction.positive_output_prob.sum_logprobs
        negative_output_prob = prediction.negative_output_prob.sum_logprobs
        return 1.0 if positive_output_prob > negative_output_prob else 0.0


class LogprobDifferenceEvaluator(Evaluator):
    """
    Evaluator that scores multiple choice examples by computing the average difference
    in sum-of-logprobs between the correct and incorrect outputs.
    """

    requires_probs = True

    def get_metric_name(self) -> str:
        return "logprob_diff"

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction based on difference in sum of logits."""

        # calculate difference in logprobs
        assert prediction.positive_output_prob is not None
        assert prediction.negative_output_prob is not None
        pos_logprobs = prediction.positive_output_prob.sum_logprobs
        neg_logprobs = prediction.negative_output_prob.sum_logprobs
        return pos_logprobs - neg_logprobs


class NormalizedPositiveProbabilityEvaluator(Evaluator):
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


def get_default_evaluators() -> Sequence[Evaluator]:
    return [
        MultipleChoiceAccuracyEvaluator(),
        LogprobDifferenceEvaluator(),
        NormalizedPositiveProbabilityEvaluator(),
    ]


def evaluate(
    pipeline: Pipeline,
    dataset: Iterable[ContrastivePair],
    evaluators: Sequence[Evaluator] = get_default_evaluators(),
    # these eval_hooks allow us to do custom stuff to the pipeline only during evaluation,
    # e.g. mess with the formatter to use CAA's special answer format
    eval_hooks: Sequence[EvalHook] = [],
    show_progress: bool = True,
    tqdm_desc: str = "Evaluating",
) -> EvalResult:
    predictions: list[EvalPrediction] = []

    # Main logic: calculate logprobs
    with ExitStack() as stack:
        for eval_hook in eval_hooks:
            stack.enter_context(eval_hook(pipeline))
        # TODO: support batching
        for i, example in enumerate(
            tqdm(dataset, disable=not show_progress, desc=tqdm_desc)
        ):
            positive_probs = pipeline.logprobs(example.positive_completion)
            negative_probs = pipeline.logprobs(example.negative_completion)

            pred = EvalPrediction(
                positive_output_prob=positive_probs,
                negative_output_prob=negative_probs,
                metrics={},
            )
            predictions.append(pred)

    # Compute metrics
    for pred in predictions:
        example_metrics = {}
        for evaluator in evaluators:
            example_metrics[evaluator.get_metric_name()] = evaluator.score_prediction(
                pred
            )
        pred.metrics = example_metrics
        predictions.append(pred)

    dataset_metrics: dict[str, float] = {}
    for evaluator in evaluators:
        dataset_metrics.update(evaluator(predictions))
    return EvalResult(predictions, dataset_metrics)
