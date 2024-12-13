"""Test that we can build all necessary datasets"""

from steering_bench.dataset import list_datasets, build_dataset, DatasetSpec


def test_list_datasets():
    datasets = list_datasets()
    assert len(datasets) > 100


def test_build_datasets():
    datasets = list_datasets()
    for dataset_name in datasets:
        spec = DatasetSpec(name=dataset_name)
        dataset = build_dataset(spec)
        assert len(dataset) > 0
