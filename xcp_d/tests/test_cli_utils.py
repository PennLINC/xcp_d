"""Tests for the xcp_d.cli.parser_utils module."""

from argparse import ArgumentParser, ArgumentTypeError
from json import JSONDecodeError
from pathlib import Path

import pytest

from xcp_d.cli import parser_utils


def test_int_or_auto():
    """Test parser_utils._int_or_auto."""
    with pytest.raises(
        ArgumentTypeError,
        match="Argument must be a nonnegative integer or 'auto'.",
    ):
        parser_utils._int_or_auto("hello")

    with pytest.raises(ArgumentTypeError, match="Int argument must be nonnegative."):
        parser_utils._int_or_auto(-2)

    out = parser_utils._int_or_auto("auto")
    assert out == "auto"

    out = parser_utils._int_or_auto("3")
    assert out == 3

    out = parser_utils._int_or_auto(3)
    assert out == 3


def test_float_or_auto():
    """Test parser_utils._float_or_auto."""
    with pytest.raises(
        ArgumentTypeError,
        match="Argument must be a nonnegative float or 'auto'.",
    ):
        parser_utils._float_or_auto("hello")

    with pytest.raises(ArgumentTypeError, match="Float argument must be nonnegative."):
        parser_utils._float_or_auto(-2)

    out = parser_utils._float_or_auto("auto")
    assert out == "auto"

    out = parser_utils._float_or_auto("3")
    assert out == 3.0

    out = parser_utils._float_or_auto(3)
    assert out == 3.0


def test_restricted_float():
    """Test parser_utils._restricted_float."""
    with pytest.raises(ArgumentTypeError, match="not a floating-point literal"):
        parser_utils._restricted_float("hello")

    with pytest.raises(ArgumentTypeError, match="not in range"):
        parser_utils._restricted_float(1.5)

    out = parser_utils._restricted_float("0.5")
    assert out == 0.5

    out = parser_utils._restricted_float(0.5)
    assert out == 0.5


def test_path_exists():
    """Test parser_utils._path_exists with an existing path."""
    parser = ArgumentParser()
    path = "/path/to/existing/file.txt"
    result = parser_utils._path_exists(path, parser)
    assert isinstance(result, Path)
    assert result == Path(path).absolute()


def test_path_exists_nonexistent():
    """Test parser_utils._path_exists with a nonexistent path."""
    parser = ArgumentParser()
    path = "/path/to/nonexistent/file.txt"
    with pytest.raises(SystemExit, match="2"):
        parser_utils._path_exists(path, parser)


def test_bids_filter_existing_path():
    """Test parser_utils._bids_filter with an existing path."""
    parser = ArgumentParser()
    value = "/path/to/existing/file.json"
    result = parser_utils._bids_filter(value, parser)
    assert isinstance(result, dict)


def test_bids_filter_nonexistent_path():
    """Test parser_utils._bids_filter with a nonexistent path."""
    parser = ArgumentParser()
    value = "/path/to/nonexistent/file.json"
    with pytest.raises(SystemExit, match="2"):
        parser_utils._bids_filter(value, parser)


def test_bids_filter_invalid_json():
    """Test parser_utils._bids_filter with an invalid JSON file."""
    parser = ArgumentParser()
    value = "/path/to/invalid/json.json"
    Path(value).write_text("invalid json")
    with pytest.raises(JSONDecodeError):
        parser_utils._bids_filter(value, parser)

    Path(value).unlink()
