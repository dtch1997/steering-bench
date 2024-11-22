import random
import pathlib

from typing import TypeVar

from steering_bench.core.types import Example, Dataset, Completion
from steering_bench.dataset.spec import DatasetSplit, DatasetSpec
from steering_bench.utils.io import jload
from steering_bench.utils.path import dataset_dir


T = TypeVar("T")

def _parse_split(split_string: str, length: int) -> slice:
    return DatasetSplit.from_str(split_string).as_slice(length)

def _get_processed_dataset_paths() -> dict[str, pathlib.Path]:
    datasets: dict[str, pathlib.Path] = {}
    for path in dataset_dir.glob("**/*.json"):
        datasets[path.stem] = path.absolute()
    return datasets

def _load_processed_dataset(dataset_path: pathlib.Path) -> Dataset:
    example_dict_list = jload(dataset_path)
    dataset: Dataset = []
    for example_dict in example_dict_list:
        dataset.append(
            Example(
                positive=Completion(**example_dict["positive"]),
                negative=Completion(**example_dict["negative"]),
                meta=example_dict.get("meta", {}),
                steering_token_index=example_dict.get("steering_token_index", -1),
            )
        )
    return dataset


def _shuffle_and_split(items: list[T], split_string: str, seed: int) -> list[T]:
    shuffled_items = items.copy()
    randgen = random.Random(seed)
    randgen.shuffle(shuffled_items)  # in-place shuffle
    split = _parse_split(split_string, len(shuffled_items))
    return shuffled_items[split]

def build_dataset(spec: DatasetSpec):
    dataset_path = _get_processed_dataset_paths()[spec.name]
    dataset = _load_processed_dataset(dataset_path)
    return _shuffle_and_split(dataset, spec.split, spec.seed)

def list_datasets() -> tuple[str, ...]:
    return tuple(_get_processed_dataset_paths().keys())