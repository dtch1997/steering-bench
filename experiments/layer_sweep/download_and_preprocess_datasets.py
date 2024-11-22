""" Script to download all datasets """

from steering_bench.dataset.download import download_persona, download_xrisk
from steering_bench.dataset.preprocess import preprocess_persona, preprocess_xrisk

if __name__ == "__main__":
    download_persona()
    download_xrisk()
    preprocess_persona()
    preprocess_xrisk()
