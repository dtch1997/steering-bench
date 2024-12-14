from inspect_ai.dataset import json_dataset, Sample

from functools import lru_cache
from steering_bench.core.types import Dataset
from steering_bench.utils.constants import DATASET_DIR
from steering_bench.utils.io import read_jsonl, write_jsonl
from steering_bench.utils.github import (
    download_github_file,
    generate_github_url,
    list_github_directory,
)
from steering_bench.dataset.mwe.preprocess import preprocess_and_randomize_dataset

DATASET_GROUP = "mwe_xrisk"


def _download_dataset(dataset_stem: str) -> tuple[bool, str]:
    github_url = generate_github_url(
        "anthropics",
        "evals",
        branch="main",
        filepath=f"advanced-ai-risk/human_generated_evals/{dataset_stem}.jsonl",
        raw=True,
    )
    status, msg = download_github_file(
        github_url, DATASET_DIR / DATASET_GROUP / f"{dataset_stem}.jsonl"
    )
    return status, msg


def record_to_sample(record: dict) -> Sample:
    breakpoint()
    return Sample(input=record["question"], target=record["answer_matching_behavior"])


@lru_cache(maxsize=1)
def list_mwe_xrisk_datasets() -> list[str]:
    infos = list_github_directory(
        "anthropics",
        "evals",
        branch="main",
        directory="advanced-ai-risk/human_generated_evals",
    )
    names = []
    for info in infos:
        if not info["name"].endswith(".jsonl"):
            continue
        filepath = info["name"]
        dataset_name = filepath.replace(".jsonl", "")
        names.append(DATASET_GROUP + "/" + dataset_name)
    return names


def build_mwe_xrisk_dataset(
    dataset_name: str,
) -> Dataset:
    # NOTE: dataset_name is of the form "mwe_v2_xrisk/<dataset_name>"
    dataset_stem = dataset_name.replace(f"{DATASET_GROUP}/", "")
    del dataset_name

    status, msg = _download_dataset(dataset_stem)
    if not status:
        raise RuntimeError(msg)

    # Preprocess the dataset to remove spurious correlations in A / B
    raw_dataset = read_jsonl(DATASET_DIR / DATASET_GROUP / f"{dataset_stem}.jsonl")
    preprocessed_dataset = preprocess_and_randomize_dataset(raw_dataset)
    write_jsonl(
        preprocessed_dataset,
        DATASET_DIR / DATASET_GROUP / f"{dataset_stem}_processed.jsonl",
    )

    dataset = json_dataset(
        # NOTE: "mwe_v2_xrisk/" prefix is already present in the dataset name
        str(DATASET_DIR / DATASET_GROUP / f"{dataset_stem}_processed.jsonl"),
        sample_fields=record_to_sample,
    )
    return dataset
