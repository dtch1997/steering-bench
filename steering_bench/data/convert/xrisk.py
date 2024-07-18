"""Script to make datasets for Anthropic's Advanced AI Risk Model-Written Evals"""

import json
import re

from steering_bench.utils.io import jdump
from steering_bench.data import Dataset
from steering_bench.data.utils import build_dataset_filename
from steering_bench.config import RawDatasetDir, PreprocessedDatasetDir
from steering_bench.core.types import ContrastivePair


def _strip_meta_tags(text: str) -> str:
    """Strip meta tags from text"""
    return re.sub(r"<META_START>[^<]*<META_END>", "", text)


def _convert_xrisk_dataset(dataset_json: list[dict[str, str]]) -> Dataset:
    """Convert a dataset in MWE format to our format"""
    dataset: Dataset = []
    for element in dataset_json:
        prompt = element["question"]
        prompt = _strip_meta_tags(prompt)
        dataset.append(
            ContrastivePair(
                prompt=prompt,
                positive_response=element["answer_matching_behavior"],
                negative_response=element["answer_not_matching_behavior"],
            )
        )
    return dataset


def convert_all_xrisk_datasets():
    """Make all datasets for Anthropic's Advanced AI Risk Model-Written Evals"""
    for dataset_path in RawDatasetDir.glob("xrisk/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        dataset_name = dataset_path.stem
        filename = build_dataset_filename(dataset_name)
        mwe_dataset: Dataset = _convert_xrisk_dataset(list_dataset)
        jdump(mwe_dataset, PreprocessedDatasetDir / "xrisk" / filename)
