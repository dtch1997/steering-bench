from dataclasses import dataclass


@dataclass
class DatasetSplit:
    """Represents a split of a dataset"""

    first_element: str
    second_element: str

    @staticmethod
    def from_str(split_str: str) -> "DatasetSplit":
        first, second = split_str.split(":")
        if first == "":
            first = "0"
        return DatasetSplit(first, second)

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

    def as_slice(self, length: int) -> slice:
        return slice(self._parse_start_idx(length), self._parse_end_idx(length))


@dataclass
class DatasetSpec:
    name: str
    split: str = ":100%"
    seed: int = 0

    def __repr__(self) -> str:
        return f"DatasetSpec(name={self.name},split={self.split},seed={self.seed})"
