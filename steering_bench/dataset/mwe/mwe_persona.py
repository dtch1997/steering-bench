from inspect_ai.dataset import json_dataset, Sample

from functools import lru_cache
from steering_bench.core.types import Dataset
from steering_bench.utils.constants import RAW_DATASET_DIR
from steering_bench.utils.github import (
    download_github_file,
    generate_github_url,
    list_github_directory,
)

DATASET_GROUP = "mwe_persona"


def _download_dataset(dataset_stem: str) -> tuple[bool, str]:
    github_url = generate_github_url(
        "anthropics",
        "evals",
        branch="main",
        filepath=f"persona/{dataset_stem}.jsonl",
        raw=True,
    )
    status, msg = download_github_file(
        github_url, RAW_DATASET_DIR / DATASET_GROUP / f"{dataset_stem}.jsonl"
    )
    return status, msg


def record_to_sample(record: dict) -> Sample:
    breakpoint()
    return Sample(input=record["question"], target=record["answer_matching_behavior"])


@lru_cache(maxsize=1)
def list_mwe_persona_datasets() -> list[str]:
    infos = list_github_directory(
        "anthropics",
        "evals",
        branch="main",
        directory="persona",
    )
    names = []
    for info in infos:
        if not info["name"].endswith(".jsonl"):
            continue
        filepath = info["name"]
        dataset_name = filepath.replace(".jsonl", "")
        names.append(DATASET_GROUP + "/" + dataset_name)
    return names


def build_mwe_persona_dataset(
    dataset_name: str,
) -> Dataset:
    # NOTE: dataset_name is of the form "mwe_v2_xrisk/<dataset_name>"
    dataset_stem = dataset_name.replace(f"{DATASET_GROUP}/", "")
    del dataset_name

    status, msg = _download_dataset(dataset_stem)
    if not status:
        raise RuntimeError(msg)

    dataset = json_dataset(
        # NOTE: "mwe_v2_xrisk/" prefix is already present in the dataset name
        str(RAW_DATASET_DIR / DATASET_GROUP / f"{dataset_stem}.jsonl"),
        sample_fields=record_to_sample,
    )
    return dataset
