from functools import partial

from steering_bench.dataset.mwe_v2.mwe_v2 import (
    build_mwe_v2_xrisk_dataset,
    list_mwe_v2_xrisk_datasets,
)
from steering_bench.dataset.registry import register_dataset

datasets = list_mwe_v2_xrisk_datasets()
for dataset_name in datasets:
    register_dataset(dataset_name, partial(build_mwe_v2_xrisk_dataset, dataset_name))
