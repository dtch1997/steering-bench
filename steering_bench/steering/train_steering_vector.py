from steering_bench.core.pipeline import Pipeline

from steering_bench.data.load import Dataset
from steering_vectors import SteeringVectorTrainingSample


def build_steering_vector_training_data(
    pipeline: Pipeline,
    dataset: Dataset,
    steering_token_index: int = -1,
) -> list[SteeringVectorTrainingSample]:
    steering_vector_training_data = [
        SteeringVectorTrainingSample(
            positive_str=pipeline.build_full_prompt(example.positive_completion),
            negative_str=pipeline.build_full_prompt(example.negative_completion),
            read_positive_token_index=steering_token_index,
            read_negative_token_index=steering_token_index,
        )
        for example in dataset
    ]
    return steering_vector_training_data


# def train_steering_vector(
#     pipeline: Pipeline,
#     dataset: Dataset,
#     layer: int,
#     steering_token_index: int = -1,
#     show_progress: bool = True,
# ):

#     steering_vector_training_data = build_steering_vector_training_data(
#         pipeline,
#         dataset,
#         steering_token_index=steering_token_index,
#     )

#     return _train_steering_vector(
#         pipeline.model,
#         pipeline.tokenizer,
#         steering_vector_training_data,
#         layers=[layer],
#         show_progress=show_progress,
#     )
