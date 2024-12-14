from functools import partial

from steering_bench.dataset.mwe.mwe_persona import (
    build_mwe_persona_dataset,
    list_mwe_persona_datasets,
)
from steering_bench.dataset.mwe.mwe_xrisk import (
    build_mwe_xrisk_dataset,
    list_mwe_xrisk_datasets,
)
from steering_bench.dataset.registry import register_dataset

datasets = list_mwe_persona_datasets()
for dataset_name in datasets:
    register_dataset(dataset_name, partial(build_mwe_persona_dataset, dataset_name))

datasets = list_mwe_xrisk_datasets()
for dataset_name in datasets:
    register_dataset(dataset_name, partial(build_mwe_xrisk_dataset, dataset_name))
