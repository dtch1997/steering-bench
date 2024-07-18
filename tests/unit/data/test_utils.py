import pytest
from steering_bench.data.utils import (
    _parse_split,
    _shuffle_and_split,
    build_dataset_filename,
    parse_dataset_filename,
)


def test_parse_split() -> None:
    assert _parse_split("0%:100%", 10) == slice(0, 10)
    assert _parse_split("0%:50%", 10) == slice(0, 5)
    assert _parse_split("50%:100%", 10) == slice(5, 10)
    assert _parse_split(":50%", 10) == slice(0, 5)
    assert _parse_split(":100%", 10) == slice(0, 10)


def test_parse_split_with_offset() -> None:
    assert _parse_split("0%:+100%", 10) == slice(0, 10)
    assert _parse_split("0%:+50%", 10) == slice(0, 5)
    assert _parse_split("50%:+50%", 10) == slice(5, 10)
    assert _parse_split(":+50%", 10) == slice(0, 5)
    assert _parse_split(":+100%", 10) == slice(0, 10)


def test_parse_split_errors_for_invalid_splits() -> None:
    with pytest.raises(ValueError):
        _parse_split("0:50%0", 10)
    with pytest.raises(ValueError):
        _parse_split("50%", 10)


def test_shuffle_and_split_returns_the_same_split_for_the_same_seed() -> None:
    split1 = _shuffle_and_split([1, 2, 3, 4], "0:50%", seed=0)
    split2 = _shuffle_and_split([1, 2, 3, 4], "0:50%", seed=0)
    assert split1 == split2
    assert len(split1) == 2


def test_shuffle_and_split_covers_all_items() -> None:
    items = [1, 2, 3, 4, 5]
    split1 = _shuffle_and_split(items, ":50%", seed=0)
    split2 = _shuffle_and_split(items, "50%:100%", seed=0)
    assert len(split1) + len(split2) == len(items)
    assert set(split1).union(set(split2)) == set(items)


def test_shuffle_and_split_leaves_original_item_unchanged() -> None:
    items = [1, 2, 3, 4, 5]
    _shuffle_and_split(items, ":50%", seed=0)
    assert items == [1, 2, 3, 4, 5]


def test_parse_dataset_filename_with_no_lang() -> None:
    filename = "blah/test.json"
    params = parse_dataset_filename(filename)
    assert params.base_name == "test"
    assert params.extension == ".json"
    assert params.lang_or_style is None


@pytest.mark.xfail(reason="Not implemented yet")
def test_parse_dataset_filename_with_lang() -> None:
    filename = "blah/test--l-fr.json"
    params = parse_dataset_filename(filename)
    assert params.base_name == "test"
    assert params.extension == ".json"
    assert params.lang_or_style == "fr"


def test_build_dataset_filename_with_no_lang() -> None:
    assert build_dataset_filename("test") == "test.json"
    assert build_dataset_filename("test", ".json") == "test.json"


@pytest.mark.xfail(reason="Not implemented yet")
def test_build_dataset_filename_with_lang() -> None:
    assert build_dataset_filename("test", ".json", "fr") == "test--l-fr.json"
    assert build_dataset_filename("test", "json", "fr") == "test--l-fr.json"
    assert build_dataset_filename("test", lang_or_style="fr") == "test--l-fr.json"


def test_build_dataset_filename_errors_on_invalid_lang() -> None:
    with pytest.raises(ValueError):
        build_dataset_filename("test", lang_or_style="FAKELANG")
