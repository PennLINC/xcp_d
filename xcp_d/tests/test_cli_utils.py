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
        parser_utils._int_or_auto('hello')

    with pytest.raises(ArgumentTypeError, match='Int argument must be nonnegative.'):
        parser_utils._int_or_auto(-2)

    out = parser_utils._int_or_auto('auto')
    assert out == 'auto'

    out = parser_utils._int_or_auto('3')
    assert out == 3

    out = parser_utils._int_or_auto(3)
    assert out == 3


def test_float_or_auto():
    """Test parser_utils._float_or_auto."""
    with pytest.raises(
        ArgumentTypeError,
        match="Argument must be a nonnegative float or 'auto'.",
    ):
        parser_utils._float_or_auto('hello')

    with pytest.raises(ArgumentTypeError, match='Float argument must be nonnegative.'):
        parser_utils._float_or_auto(-2)

    out = parser_utils._float_or_auto('auto')
    assert out == 'auto'

    out = parser_utils._float_or_auto('3')
    assert out == 3.0

    out = parser_utils._float_or_auto(3)
    assert out == 3.0


def test_restricted_float():
    """Test parser_utils._restricted_float."""
    with pytest.raises(ArgumentTypeError, match='not a floating-point literal'):
        parser_utils._restricted_float('hello')

    with pytest.raises(ArgumentTypeError, match='not in range'):
        parser_utils._restricted_float(1.5)

    out = parser_utils._restricted_float('0.5')
    assert out == 0.5

    out = parser_utils._restricted_float(0.5)
    assert out == 0.5


def test_float_or_auto_or_none():
    """Test parser_utils._float_or_auto_or_none."""
    with pytest.raises(
        ArgumentTypeError,
        match="Argument must be a nonnegative float, 'all', or 'none', not 'hello'.",
    ):
        parser_utils._float_or_auto_or_none('hello')

    with pytest.raises(ArgumentTypeError, match='Float argument must be nonnegative.'):
        parser_utils._float_or_auto_or_none(-2)

    out = parser_utils._float_or_auto_or_none('all')
    assert out == 'all'

    out = parser_utils._float_or_auto_or_none('none')
    assert out == 'none'

    out = parser_utils._float_or_auto_or_none('3')
    assert out == '3.0'

    out = parser_utils._float_or_auto_or_none(3)
    assert out == '3.0'


def test_is_file(tmp_path_factory):
    """Test parser_utils._is_file."""
    tmpdir = tmp_path_factory.mktemp('test_is_file')

    # Existing file
    with open(tmpdir / 'file.txt', 'w') as f:
        f.write('')

    parser = ArgumentParser()
    result = parser_utils._is_file(str(tmpdir / 'file.txt'), parser)
    assert isinstance(result, Path)
    assert result == Path(tmpdir / 'file.txt').absolute()

    # Nonexistent file
    parser = ArgumentParser()
    path = '/path/to/nonexistent/file.txt'
    with pytest.raises(SystemExit, match='2'):
        parser_utils._is_file(path, parser)

    # Path, not a file
    parser = ArgumentParser()
    with pytest.raises(SystemExit, match='2'):
        parser_utils._is_file(str(tmpdir), parser)


def test_path_exists(tmp_path_factory):
    """Test parser_utils._path_exists."""
    tmpdir = tmp_path_factory.mktemp('test_path_exists')

    # Existing path
    parser = ArgumentParser()
    result = parser_utils._path_exists(str(tmpdir), parser)
    assert isinstance(result, Path)
    assert result == Path(tmpdir).absolute()

    # Nonexistent path
    parser = ArgumentParser()
    path = '/path/to/nonexistent/file.txt'
    with pytest.raises(SystemExit, match='2'):
        parser_utils._path_exists(path, parser)


def test_bids_filter(tmp_path_factory):
    """Test parser_utils._bids_filter."""
    tmpdir = tmp_path_factory.mktemp('test_bids_filter_existing_path')

    # Existing path with valid JSON
    json_file = str(tmpdir / 'file.json')
    with open(json_file, 'w') as f:
        f.write('{}')

    parser = ArgumentParser()
    result = parser_utils._bids_filter(json_file, parser)
    assert isinstance(result, dict)

    # Nonexistent path
    parser = ArgumentParser()
    value = '/path/to/nonexistent/file.json'
    with pytest.raises(SystemExit, match='2'):
        parser_utils._bids_filter(value, parser)

    # Invalid JSON
    tmpdir = tmp_path_factory.mktemp('test_bids_filter_invalid_json')
    json_file = str(tmpdir / 'invalid.json')
    with open(json_file, 'w') as f:
        f.write('invalid json')

    parser = ArgumentParser()
    with pytest.raises(SystemExit, match='2'):
        parser_utils._bids_filter(json_file, parser)

    # No value
    parser = ArgumentParser()
    result = parser_utils._bids_filter(None, parser)
    assert result is None


def test_yes_no_action():
    """Test parser_utils.YesNoAction."""
    parser = ArgumentParser()
    parser.add_argument('--option', nargs='?', action=parser_utils.YesNoAction)

    # A value of y should be True
    args = parser.parse_args(['--option', 'y'])
    assert args.option is True

    # A value of n should be False
    args = parser.parse_args(['--option', 'n'])
    assert args.option is False

    # The parameter without a value should default to True
    args = parser.parse_args(['--option'])
    assert args.option is True

    # Auto is an option
    args = parser.parse_args(['--option', 'auto'])
    assert args.option == 'auto'

    # Invalid value raises an error
    with pytest.raises(SystemExit):
        parser.parse_args(['--option', 'invalid'])


def test_to_dict():
    """Test parser_utils.ToDict."""
    parser = ArgumentParser()
    parser.add_argument('--option', action=parser_utils.ToDict, nargs='+')

    # Two key-value pairs
    args = parser.parse_args(['--option', 'key1=value1', 'key2=value2'])
    assert args.option == {'key1': Path('value1'), 'key2': Path('value2')}

    # Providing the same key twice
    with pytest.raises(SystemExit):
        parser.parse_args(['--option', 'key1=value1', 'key1=value2'])

    # Trying to use one of the reserved keys
    with pytest.raises(SystemExit):
        parser.parse_args(['--option', 'preprocessed=value1'])

    # Dataset with no name
    args = parser.parse_args(['--option', 'value1'])
    assert args.option == {'value1': Path('value1')}


def test_confounds_action(tmp_path):
    """Test parser_utils.ConfoundsAction."""
    parser = ArgumentParser()
    parser.add_argument('--confounds', action=parser_utils.ConfoundsAction)

    # A value of auto should be "auto"
    args = parser.parse_args(['--confounds', 'auto'])
    assert args.confounds == 'auto'

    # A valid custom confounds option
    valid_path = tmp_path / 'valid_confounds.yml'
    valid_path.touch()  # Create the file
    args = parser.parse_args(['--confounds', str(valid_path)])
    assert args.confounds == valid_path

    # Path to a non-existent file should raise an error
    with pytest.raises(SystemExit):
        parser.parse_args(['--confounds', '/invalid/path/to/confounds.yml'])


def test_task_id_argument(tmp_path_factory):
    """Test that --task-id accepts space-delimited list and removes 'task-' prefix."""
    from xcp_d.cli.parser import _build_parser

    # Create temporary directories for input and output
    tmpdir = tmp_path_factory.mktemp('test_task_id')
    input_dir = tmpdir / 'input'
    output_dir = tmpdir / 'output'
    input_dir.mkdir()
    output_dir.mkdir()

    parser = _build_parser()
    base_args = [str(input_dir), str(output_dir), 'participant', '--mode', 'none']

    # Single task-id without prefix
    args = parser.parse_args(base_args + ['--task-id', 'rest'])
    assert args.task_id == ['rest']

    # Single task-id with prefix
    args = parser.parse_args(base_args + ['--task-id', 'task-rest'])
    assert args.task_id == ['rest']

    # Multiple task-ids without prefix
    args = parser.parse_args(base_args + ['--task-id', 'rest', 'imagery', 'nback'])
    assert args.task_id == ['rest', 'imagery', 'nback']

    # Multiple task-ids with prefix
    args = parser.parse_args(base_args + ['--task-id', 'task-rest', 'task-imagery'])
    assert args.task_id == ['rest', 'imagery']

    # Mixed prefix usage
    args = parser.parse_args(base_args + ['--task-id', 'rest', 'task-imagery'])
    assert args.task_id == ['rest', 'imagery']
