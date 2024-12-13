"""Preprocessing logic for datasets"""

import json
import re
import pathlib

from steering_bench.core.types import Dataset, Completion, Example
from steering_bench.utils.path import raw_dataset_dir, assets_dir
from steering_bench.utils.io import jdump


def get_all_files_in_dir(
    dir: pathlib.Path,
    extension: str = ".jsonl",
) -> list[pathlib.Path]:
    return list(dir.glob(f"*{extension}" if extension else "*"))


def _maybe_swap(a, b, condition):
    if condition:
        return b, a
    return a, b


def preprocess_and_randomize_dataset(
    list_data: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Preprocess and randomize the dataset

    Randomization ensures that there are an equal number of samples where the correct answer is A and B respectively.
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


def strip_meta_tags(text: str) -> str:
    """Strip meta tags from text"""
    return re.sub(r"<META_START>[^<]*<META_END>", "", text)


def convert_xrisk_dataset(mwe: list[dict[str, str]]) -> Dataset:
    """Convert a MWE XRisk dataset to our format"""
    mwe_dataset: Dataset = []
    for element in mwe:
        prompt = element["question"]
        prompt = strip_meta_tags(prompt)
        positive = Completion(
            prompt=prompt,
            response=element["answer_matching_behavior"],
        )

        negative = Completion(
            prompt=prompt,
            response=element["answer_not_matching_behavior"],
        )
        mwe_dataset.append(
            Example(
                positive=positive,
                negative=negative,
                steering_token_index=-2,
            )
        )
    return mwe_dataset


def convert_persona_dataset(
    raw_dataset: list[dict[str, str]],
) -> Dataset:
    """Convert a MWE Persona dataset to our format"""
    dataset: Dataset = []
    for element in preprocess_and_randomize_dataset(raw_dataset):
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


def preprocess_persona():
    """Make MWE dataset"""
    for dataset_path in get_all_files_in_dir(raw_dataset_dir / "mwe_persona"):
        # Load the jsonl file
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        dataset = convert_persona_dataset(list_dataset)
        dataset_name = dataset_path.stem
        jdump(
            dataset,
            assets_dir / "processed_datasets" / "mwe_persona" / f"{dataset_name}.json",
        )


def preprocess_xrisk():
    """Make MWE dataset"""
    for dataset_path in get_all_files_in_dir(raw_dataset_dir / "mwe_xrisk"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        dataset_name = dataset_path.stem
        xrisk_dataset: Dataset = convert_xrisk_dataset(list_dataset)
        jdump(
            xrisk_dataset,
            assets_dir / "processed_datasets" / "mwe_xrisk" / f"{dataset_name}.json",
        )
