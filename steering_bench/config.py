"""Define configuration variables"""

import pathlib

ProjectDir = pathlib.Path(__file__).resolve().parents[1].absolute()
PackageDir = ProjectDir / "steering_bench"
DatasetDir = PackageDir / "datasets"
RawDatasetDir = DatasetDir / "raw"
PreprocessedDatasetDir = DatasetDir / "preprocessed"
