import pathlib

project_dir = pathlib.Path(__file__).resolve().parents[2]
assets_dir = project_dir / "assets"
raw_dataset_dir = project_dir / "assets" / "raw_datasets"
dataset_dir = project_dir / "assets" / "processed_datasets"
