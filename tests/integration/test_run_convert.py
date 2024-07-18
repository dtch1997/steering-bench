import pathlib

from steering_bench.config import RawDatasetDir, PreprocessedDatasetDir
from steering_bench.data.run_convert import convert_all_datasets


def get_dataset_name(relative_path: pathlib.Path, extension=".json") -> str:
    return str(relative_path).replace(extension, "")


def test_all_datasets_are_converted():
    convert_all_datasets()

    all_raw_dataset_paths = list(RawDatasetDir.glob("**/*.jsonl"))
    all_raw_dataset_names = [
        get_dataset_name(path.relative_to(RawDatasetDir), extension=".jsonl")
        for path in all_raw_dataset_paths
    ]
    assert len(all_raw_dataset_names) > 0, "No raw datasets found"

    all_preprocessed_dataset_paths = list(PreprocessedDatasetDir.glob("**/*.json"))
    all_preprocessed_dataset_names = [
        get_dataset_name(path.relative_to(PreprocessedDatasetDir))
        for path in all_preprocessed_dataset_paths
    ]
    assert len(all_preprocessed_dataset_names) > 0, "No preprocessed datasets found"

    missing_datasets = set(all_raw_dataset_names) - set(all_preprocessed_dataset_names)
    assert (
        missing_datasets == set()
    ), f"Missing {len(missing_datasets)} datasets: {missing_datasets}"
