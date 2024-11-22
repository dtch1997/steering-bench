import requests
import io
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from steering_bench.core.types import Dataset, Example, Completion
from steering_bench.utils.io import jdump, jload
from steering_bench.utils.path import raw_dataset_dir, dataset_dir


def download_harmful_instructions():
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)

    jdump(train, raw_dataset_dir / "refusal" / "harmful_instructions_train.json")
    jdump(test, raw_dataset_dir / "refusal" / "harmful_instructions_test.json")


def download_harmless_instructions():
    hf_path = "tatsu-lab/alpaca"
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset["train"])):  # type: ignore
        if dataset["train"][i]["input"].strip() == "":  # type: ignore
            instructions.append(dataset["train"][i]["instruction"])  # type: ignore

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    jdump(train, raw_dataset_dir / "refusal" / "harmless_instructions_train.json")
    jdump(test, raw_dataset_dir / "refusal" / "harmless_instructions_test.json")


def convert_train_dataset(
    harmful_instructions: list[str],
    harmless_instructions: list[str],
) -> Dataset:
    """Convert the raw dataset into a format that can be used by the steering bench

    Training needs contrast between the harmful and harmless instructions, so we need to pair them up
    """
    examples = []
    for harmful_inst, harmless_inst in zip(harmful_instructions, harmless_instructions):
        # The positive behaviour is refusal
        positive = Completion(
            prompt=harmful_inst,
            response="",
        )
        negative = Completion(
            prompt=harmless_inst,
            response="",
        )
        example = Example(
            positive=positive,
            negative=negative,
            steering_token_index=-1,
        )
        examples.append(example)
    return examples


def convert_test_dataset(
    instructions: list[str],
) -> Dataset:
    """Convert the raw dataset into a format that can be used by the steering bench

    When testing, we only need a single instruction
    """
    examples = []
    for inst in instructions:
        # The positive behaviour is refusal
        positive = Completion(
            prompt=inst,
            response="I",
        )
        negative = Completion(
            prompt=inst,
            response="Sure",
        )
        example = Example(
            positive=positive,
            negative=negative,
            steering_token_index=-1,
        )
        examples.append(example)
    return examples


def preprocess_train_dataset():
    harmful_insts = jload(
        raw_dataset_dir / "refusal" / "harmful_instructions_train.json"
    )
    harmless_insts = jload(
        raw_dataset_dir / "refusal" / "harmless_instructions_train.json"
    )

    # process a combined dataset
    dataset = convert_train_dataset(harmful_insts, harmless_insts)
    jdump(dataset, dataset_dir / "refusal" / "refusal_train.json")


def preprocess_test_datasets():
    harmful_insts = jload(
        raw_dataset_dir / "refusal" / "harmful_instructions_test.json"
    )
    dataset = convert_test_dataset(harmful_insts)
    jdump(dataset, dataset_dir / "refusal" / "harmful_instructions_test.json")

    harmless_insts = jload(
        raw_dataset_dir / "refusal" / "harmless_instructions_test.json"
    )
    dataset = convert_test_dataset(harmless_insts)
    jdump(dataset, dataset_dir / "refusal" / "harmless_instructions_test.json")


if __name__ == "__main__":
    download_harmful_instructions()
    download_harmless_instructions()
    preprocess_train_dataset()
    preprocess_test_datasets()
