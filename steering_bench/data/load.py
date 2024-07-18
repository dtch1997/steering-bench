import pathlib

from dataclasses import dataclass

from steering_bench.core.types import ContrastivePair
from steering_bench.config import DatasetDir
from steering_bench.utils.io import jload
from steering_bench.data.utils import _shuffle_and_split

Dataset = list[ContrastivePair]

@dataclass
class DatasetSpec:
    name: str
    split: str = ":100%"
    seed: int = 0

    def __repr__(self) -> str:
        return f"DatasetSpec(name={self.name},split={self.split},seed={self.seed})"
    

def _get_all_json_filepaths(root_dir: pathlib.Path) -> list[pathlib.Path]:
    return list(root_dir.rglob("*.json"))


def _get_available_datasets(dataset_dir: pathlib.Path | None = DatasetDir) -> dict[str, pathlib.Path]:
    datasets: dict[str, pathlib.Path] = {}
    for path in _get_all_json_filepaths(dataset_dir):
        # NOTE: using path.stem doesn't handle nested directories very well.
        # dataset_name = path.stem        

        # Get the relative path from the dataset_dir and use that as the key 
        dataset_name = path.relative_to(dataset_dir).with_suffix("").as_posix()
        datasets[dataset_name] = path.absolute()
    return datasets

def _load_dataset(filepath: pathlib.Path) -> Dataset:
    """ Load a dataset from a json file. """
    example_dict_list = jload(filepath)
    dataset: Dataset = []
    for example_dict in example_dict_list:
        dataset.append(
            ContrastivePair(
                prompt=example_dict["prompt"],
                positive_response=example_dict["positive_response"],
                negative_response=example_dict["negative_response"],
            )
        )
    return dataset

def load_dataset(spec: DatasetSpec, dataset_dir: pathlib.Path | None = None):
    """ Standard method for loading a dataset. """

    datasets = _get_available_datasets(dataset_dir)
    if spec.name not in datasets:
        raise ValueError(f"Unknown dataset: {spec.name}")

    dataset = _load_dataset(spec.name, dataset_dir)
    return _shuffle_and_split(dataset, spec.split, spec.seed)