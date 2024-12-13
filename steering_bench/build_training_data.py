import logging

from steering_vectors import SteeringVectorTrainingSample
from steering_bench.core.types import Dataset
from steering_bench.core.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _validate_train_dataset(dataset: Dataset):
    steering_token_index = dataset[0].steering_token_index
    for example in dataset:
        assert example.steering_token_index == steering_token_index


def build_steering_vector_training_data(
    pipeline: Pipeline,
    dataset: Dataset,
) -> list[SteeringVectorTrainingSample]:
    """Build steering vector training data

    Checks that all examples have the same steering token index
    Applies the pipeline's formatting logic to build the positive and negative examples
    """
    # Validate that all examples have the same steering token index
    _validate_train_dataset(dataset)
    # After validation, we can assume that all examples have the same steering token index
    read_token_index = dataset[0].steering_token_index

    steering_vector_training_data = [
        SteeringVectorTrainingSample(
            positive_str=pipeline.build_full_prompt(example.positive),
            negative_str=pipeline.build_full_prompt(example.negative),
            read_positive_token_index=read_token_index,
            read_negative_token_index=read_token_index,
        )
        for example in dataset
    ]

    # Log first example
    datum = steering_vector_training_data[0]
    logger.debug(f"Positive example: \n {datum.positive_str}")
    logger.debug(f"Negative example: \n {datum.negative_str}")

    return steering_vector_training_data
