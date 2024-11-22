"""Utilities for reading, writing datasets to disk"""

# Taken from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/utils.py

from dataclasses import asdict, is_dataclass
import io
import json
import os

from typing import Any


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


class DataclassEncoder(json.JSONEncoder):
    """JSON encoder which can properly handle dataclasses."""

    def default(self, o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)  # type: ignore
        return json.JSONEncoder.default(self, o)


def jdump(obj, f, mode="w", indent=4):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, cls=DataclassEncoder, ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r") -> Any:
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
