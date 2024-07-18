"""Script to download and convert all datasets"""

from steering_bench.data.convert.persona import convert_all_persona_datasets
from steering_bench.data.convert.xrisk import convert_all_xrisk_datasets


def convert_all_datasets():
    convert_all_persona_datasets()
    convert_all_xrisk_datasets()


if __name__ == "__main__":
    convert_all_datasets
