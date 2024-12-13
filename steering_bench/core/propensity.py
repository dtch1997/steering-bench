import numpy as np

from steering_bench.core.evaluate import EvalPrediction, PropensityScore


class Accuracy(PropensityScore):
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


class LogProbDifference(PropensityScore):
    """
    Computes the average difference in logprob between the correct and incorrect outputs.
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


class NormalizedPositiveProbability(PropensityScore):
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
