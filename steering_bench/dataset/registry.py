from typing import Any, Callable, Dict, List, TypeVar

T = TypeVar("T")
DatasetBuilder = Callable[..., T]

_DATASET_REGISTRY: Dict[str, DatasetBuilder] = {}


def register_dataset(name: str, builder: DatasetBuilder[T]) -> DatasetBuilder[T]:
    """
    Register a dataset builder function.

    Args:
        name: Name of the dataset
        builder: Function that builds/loads the dataset

    Returns:
        The original builder function

    Raises:
        ValueError: If dataset name is already registered
    """
    if name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is already registered")

    _DATASET_REGISTRY[name] = builder
    return builder


def dataset(name: str) -> Callable[[DatasetBuilder[T]], DatasetBuilder[T]]:
    """
    Decorator to register a dataset builder function.

    Args:
        name: Name of the dataset

    Returns:
        Decorator function that registers the dataset builder
    """

    def decorator(func: DatasetBuilder[T]) -> DatasetBuilder[T]:
        return register_dataset(name, func)

    return decorator


def build_dataset(name: str, **kwargs: Any) -> Any:
    """
    Build a dataset by name using the registered builder function.

    Args:
        name: Name of the dataset to build
        **kwargs: Arguments to pass to the dataset builder

    Returns:
        The built dataset

    Raises:
        KeyError: If dataset name is not found in registry
    """
    if name not in _DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{name}' not found in registry. Available datasets: {list_datasets()}"
        )

    builder = _DATASET_REGISTRY[name]
    return builder(**kwargs)


def list_datasets() -> List[str]:
    """
    List all registered dataset names.

    Returns:
        List of registered dataset names
    """
    return sorted(list(_DATASET_REGISTRY.keys()))
