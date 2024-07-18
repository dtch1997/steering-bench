"""Script to make datasets for Anthropic's Persona Model-Written Evals"""

import json
from typing import cast

from steering_bench.data.utils import build_dataset_filename
from steering_bench.utils.io import jdump
from steering_bench.core.types import Dataset, Example, Completion
from steering_bench.config import RawDatasetDir, PreprocessedDatasetDir


def _maybe_swap(a, b, condition):
    if condition:
        return b, a
    return a, b


def preprocess_ab_dataset(
    list_data: list[dict[str, str]]
) -> list[dict[str, str]]:
    """ Processes the dataset to be A/B formatted 

    Swaps the choice of whether A or B is positive to remove correlation with the behaviour
    Swap every 2 samples to avoid undoing the alternating behavior of the MWE data.
     
    Example input:
    ```
    {
        "question": "What is the capital of France?",
        "answer_matching_behavior": "Paris",
        "answer_not_matching_behavior": "London"
    }
    ```

    Example output:
    ```
    {
        "question": "What is the capital of France?\n\nChoices:\n(A): Paris\n(B): London",
        "answer_matching_behavior": "(A)",
        "answer_not_matching_behavior": "(B)"
    }
    """
    new_list_data = []
    for i, data in enumerate(list_data):
        question = data["question"]
        answer_a = data["answer_matching_behavior"]
        answer_b = data["answer_not_matching_behavior"]

        # Construct A/B formatted question / answer pair
        choices_str = "Choices:"
        new_answer_matching_behavior = "(A)"
        new_answer_not_matching_behavior = "(B)"

        # Swap A and B every 2 samples
        # The MWE data alternates every sample, so don't want to undo their alternating
        swap = int(i / 2) % 2 == 0
        answer_a, answer_b = _maybe_swap(answer_a, answer_b, swap)
        new_question = f"{question}\n\n{choices_str}\n(A): {answer_a}\n(B): {answer_b}"
        new_answer_matching_behavior, new_answer_not_matching_behavior = _maybe_swap(
            new_answer_matching_behavior, new_answer_not_matching_behavior, swap
        )

        new_list_data.append(
            {
                "question": new_question,
                "answer_matching_behavior": new_answer_matching_behavior,
                "answer_not_matching_behavior": new_answer_not_matching_behavior,
            }
        )

    return new_list_data


def _convert_persona_dataset(
    dataset_json: list[dict[str, str]],
) -> Dataset:
    """Convert a dataset in MWE format to our format"""
    dataset: Dataset = []
    for element in preprocess_ab_dataset(dataset_json):
        prompt = element["question"]
        positive = Completion(
            prompt=prompt,
            response=element["answer_matching_behavior"],
        )

        negative = Completion(
            prompt=prompt,
            response=element["answer_not_matching_behavior"],
        )

        ex = Example(positive=positive, negative=negative, steering_token_index=-2)
        dataset.append(ex)
    return dataset


def convert_all_persona_datasets():
    """ Make all datasets for Anthropic's Persona Model-Written Evals """
    for dataset_path in RawDatasetDir.glob("persona/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        persona = dataset_path.stem
        filename = build_dataset_filename(persona)
        mwe_dataset: Dataset = _convert_persona_dataset(list_dataset)
        jdump(mwe_dataset, PreprocessedDatasetDir / "persona" / filename)