"""Tests for the xcp_d.cli.parser_utils module."""
from argparse import ArgumentTypeError

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
