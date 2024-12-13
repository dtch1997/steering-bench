from steering_bench.dataset.mwe.download import download_persona, download_xrisk
from steering_bench.dataset.mwe.preprocess import preprocess_persona, preprocess_xrisk


def run_download_and_preprocess():
    download_persona()
    download_xrisk()
    preprocess_persona()
    preprocess_xrisk()


if __name__ == "__main__":
    run_download_and_preprocess()
