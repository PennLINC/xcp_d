"""Tests for the xcp_d.cli.parser_utils module."""

from argparse import ArgumentParser, ArgumentTypeError
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


def test_float_or_auto_or_none():
    """Test parser_utils._float_or_auto_or_none."""
    with pytest.raises(
        ArgumentTypeError,
        match="Argument must be a nonnegative float, 'all', or 'none', not 'hello'.",
    ):
        parser_utils._float_or_auto_or_none("hello")

    with pytest.raises(ArgumentTypeError, match="Float argument must be nonnegative."):
        parser_utils._float_or_auto_or_none(-2)

    out = parser_utils._float_or_auto_or_none("all")
    assert out == "all"

    out = parser_utils._float_or_auto_or_none("none")
    assert out == "none"

    out = parser_utils._float_or_auto_or_none("3")
    assert out == 3.0

    out = parser_utils._float_or_auto_or_none(3)
    assert out == 3.0


def test_is_file(tmp_path_factory):
    """Test parser_utils._is_file."""
    tmpdir = tmp_path_factory.mktemp("test_is_file")

    # Existing file
    with open(tmpdir / "file.txt", "w") as f:
        f.write("")

    parser = ArgumentParser()
    result = parser_utils._is_file(str(tmpdir / "file.txt"), parser)
    assert isinstance(result, Path)
    assert result == Path(tmpdir / "file.txt").absolute()

    # Nonexistent file
    parser = ArgumentParser()
    path = "/path/to/nonexistent/file.txt"
    with pytest.raises(SystemExit, match="2"):
        parser_utils._is_file(path, parser)

    # Path, not a file
    parser = ArgumentParser()
    with pytest.raises(SystemExit, match="2"):
        parser_utils._is_file(str(tmpdir), parser)


def test_path_exists(tmp_path_factory):
    """Test parser_utils._path_exists."""
    tmpdir = tmp_path_factory.mktemp("test_path_exists")

    # Existing path
    parser = ArgumentParser()
    result = parser_utils._path_exists(str(tmpdir), parser)
    assert isinstance(result, Path)
    assert result == Path(tmpdir).absolute()

    # Nonexistent path
    parser = ArgumentParser()
    path = "/path/to/nonexistent/file.txt"
    with pytest.raises(SystemExit, match="2"):
        parser_utils._path_exists(path, parser)


def test_bids_filter(tmp_path_factory):
    """Test parser_utils._bids_filter."""
    tmpdir = tmp_path_factory.mktemp("test_bids_filter_existing_path")

    # Existing path with valid JSON
    json_file = str(tmpdir / "file.json")
    with open(json_file, "w") as f:
        f.write("{}")

    parser = ArgumentParser()
    result = parser_utils._bids_filter(json_file, parser)
    assert isinstance(result, dict)

    # Nonexistent path
    parser = ArgumentParser()
    value = "/path/to/nonexistent/file.json"
    with pytest.raises(SystemExit, match="2"):
        parser_utils._bids_filter(value, parser)

    # Invalid JSON
    tmpdir = tmp_path_factory.mktemp("test_bids_filter_invalid_json")
    json_file = str(tmpdir / "invalid.json")
    with open(json_file, "w") as f:
        f.write("invalid json")

    parser = ArgumentParser()
    with pytest.raises(SystemExit, match="2"):
        parser_utils._bids_filter(json_file, parser)

    # No value
    parser = ArgumentParser()
    result = parser_utils._bids_filter(None, parser)
    assert result is None
