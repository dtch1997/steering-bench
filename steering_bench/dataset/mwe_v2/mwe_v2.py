from inspect_ai.dataset import json_dataset, Sample

from functools import lru_cache
from steering_bench.core.types import Dataset
from steering_bench.utils.constants import RAW_DATASET_DIR
from steering_bench.utils.github import (
    download_github_file,
    generate_github_url,
    list_github_directory,
)


def _download_dataset(dataset_name: str) -> tuple[bool, str]:
    github_url = generate_github_url(
        "Sunishchal",
        "model-written-evals",
        branch="main",
        filepath=f"improved_datasets/{dataset_name}.jsonl",
        raw=True,
    )
    status, msg = download_github_file(
        github_url, RAW_DATASET_DIR / "mwe_v2_xrisk" / f"{dataset_name}.jsonl"
    )
    return status, msg


def record_to_sample(record: dict) -> Sample:
    breakpoint()
    return Sample(
        input=record["question"],
        target=record["choice_a"]
        if record["answer_matching_behavior"] == "(A)"
        else record["choice_b"],
    )


@lru_cache(maxsize=1)
def list_mwe_v2_xrisk_datasets() -> list[str]:
    infos = list_github_directory(
        "Sunishchal",
        "model-written-evals",
        branch="main",
        directory="improved_datasets",
    )
    names = []
    for info in infos:
        filepath = info["name"]
        dataset_name = filepath.rstrip(".jsonl")
        names.append("mwe_v2_xrisk/" + dataset_name)
    return names


def build_mwe_v2_xrisk_dataset(
    dataset_name: str,
) -> Dataset:
    # NOTE: dataset_name is of the form "mwe_v2_xrisk/<dataset_name>"
    dataset_name = dataset_name.replace("mwe_v2_xrisk/", "")

    status, msg = _download_dataset(dataset_name)
    if not status:
        raise RuntimeError(msg)

    raise NotImplementedError("Blocked on upstream issue")
    # NOTE: Some samples are mising 'choice_a' / 'choice_b' fields
    # As such the following code will throw an error
    # Potential workaround: Filter out samples with missing 'choice_a' / 'choice_b' fields

    dataset = json_dataset(
        # NOTE: "mwe_v2_xrisk/" prefix is already present in the dataset name
        str(RAW_DATASET_DIR / "mwe_v2_xrisk" / f"{dataset_name}.jsonl"),
        sample_fields=record_to_sample,
    )
    return dataset
