import random
import pathlib

from typing import TypeVar
from dataclasses import dataclass
from steering_bench.data.translation.constants import (
    LANG_OR_STYLE_MAPPING,
    LangOrStyleCode,
)

T = TypeVar("T")


def _parse_split(split_string: str, length: int) -> slice:
    return DatasetSplit.from_str(split_string).as_slice(length)


def _shuffle_and_split(items: list[T], split_string: str, seed: int) -> list[T]:
    randgen = random.Random(seed)
    shuffled_items = items.copy()
    randgen.shuffle(shuffled_items)  # in-place shuffle
    split = _parse_split(split_string, len(shuffled_items))
    return shuffled_items[split]


@dataclass
class DatasetSplit:
    """Represents a split of a dataset.

    Example:
    ```
    split = DatasetSplit.from_str("0:100%")
    assert split.as_slice(20) == slice(0, 20)
    ```
    """

    first_element: str
    second_element: str

    @staticmethod
    def from_str(split_str: str) -> "DatasetSplit":
        first, second = split_str.split(":")
        if first == "":
            first = "0"
        return DatasetSplit(first, second)

    def as_slice(self, length: int) -> slice:
        return slice(self._parse_start_idx(length), self._parse_end_idx(length))

    def _parse_start_idx(self, length: int) -> int:
        element = self.first_element
        if element.endswith("%"):
            k = int(element[:-1]) * length // 100
        else:
            k = int(element)

        if k < 0 or k > length:
            raise ValueError(f"Invalid index: k={k} in {self}")
        return k

    def _parse_end_idx(self, length: int) -> int:
        element = self.second_element

        should_add = element.startswith("+")
        element = element.lstrip("+")

        if element.endswith("%"):
            k = int(element[:-1]) * length // 100
        else:
            k = int(element)

        if should_add:
            k = self._parse_start_idx(length) + k

        if k < 0 or k > length:
            raise ValueError(f"Invalid index: k={k} in {self}")

        return k


@dataclass
class DatasetFilenameSpec:
    """Represents a dataset filename"""

    base_name: str
    extension: str
    lang_or_style: LangOrStyleCode | None = None


def build_dataset_filename(
    base_name: str,
    extension: str = "json",
    lang_or_style: LangOrStyleCode | str | None = None,
) -> str:
    if lang_or_style is not None and lang_or_style not in LANG_OR_STYLE_MAPPING:
        raise ValueError(f"Unknown lang_or_style: {lang_or_style}")
    if extension.startswith("."):
        extension = extension[1:]
    if lang_or_style is None:
        return f"{base_name}.{extension}"
    return f"{base_name}--l-{lang_or_style}.{extension}"


def parse_dataset_filename(filename: str | pathlib.Path) -> DatasetFilenameSpec:
    filepath = pathlib.Path(filename)

    # TODO: add back translation handling
    # if "--l-" in filepath.stem:
    #     base_name, lang_or_style = filepath.stem.split("--l-")
    #     if lang_or_style not in LANG_OR_STYLE_MAPPING:
    #         raise ValueError(f"Unknown lang_or_style: {lang_or_style}")
    #     return DatasetFilenameSpec(
    #         base_name, extension=filepath.suffix, lang_or_style=lang_or_style
    #     )

    return DatasetFilenameSpec(
        base_name=filepath.stem, extension=filepath.suffix, lang_or_style=None
    )
