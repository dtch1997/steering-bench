"""Utilities for managing torch device"""

import torch
from contextlib import contextmanager
from typing import Generator
from .patterns import Singleton


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Parse the PyTorch version to check if it's below version 2.0
        major_version = int(torch.__version__.split(".")[0])
        if major_version >= 2:
            return "mps"
    else:
        return "cpu"

    raise RuntimeError("Should not reach here!")


@Singleton
class DeviceManager:
    """Device manager class

    Example:
    ```
    with DeviceManager.instance().use_device("cuda"):
        ...
    ```
    """

    device: str

    def __init__(self) -> None:
        self.device = get_default_device()

    def get_device(self) -> str:
        return self.device

    def set_device(self, device: str) -> None:
        self.device = device

    @contextmanager
    def use_device(self, device: str) -> Generator[None, None, None]:
        old_device = self.get_device()
        self.set_device(device)
        yield
        self.set_device(old_device)
