"""Utility functions for xcp_d command-line interfaces."""

import argparse
import logging
import warnings
from argparse import Action
from pathlib import Path

warnings.filterwarnings('ignore')

logging.addLevelName(25, 'IMPORTANT')  # Add a new level between INFO and WARNING
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG
logger = logging.getLogger('cli')


def _int_or_auto(string, is_parser=True):
    """Check if argument is an integer >= 0 or the string "auto"."""
    if string == 'auto':
        return string

    error = argparse.ArgumentTypeError if is_parser else ValueError
    try:
        intarg = int(string)
    except ValueError as exc:
        msg = "Argument must be a nonnegative integer or 'auto'."
        raise error(msg) from exc

    if intarg < 0:
        raise error('Int argument must be nonnegative.')
    return intarg


def _float_or_auto(string, is_parser=True):
    """Check if argument is a float >= 0 or the string "auto"."""
    if string == 'auto':
        return string

    error = argparse.ArgumentTypeError if is_parser else ValueError
    try:
        floatarg = float(string)
    except ValueError as exc:
        msg = "Argument must be a nonnegative float or 'auto'."
        raise error(msg) from exc

    if floatarg < 0:
        raise error('Float argument must be nonnegative.')
    return floatarg


def _float_or_auto_or_none(string, is_parser=True):
    """Check if argument is a float >= 0 or the strings "all" or "none"."""
    if string in ('all', 'none'):
        return string

    error = argparse.ArgumentTypeError if is_parser else ValueError
    try:
        floatarg = float(string)
    except ValueError as exc:
        msg = f"Argument must be a nonnegative float, 'all', or 'none', not '{string}'."
        raise error(msg) from exc

    if floatarg < 0:
        raise error('Float argument must be nonnegative.')
    return str(floatarg)


def _restricted_float(x):
    """From https://stackoverflow.com/a/12117065/2589328."""
    if x == 'auto':
        return x

    try:
        x = float(x)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{x} not a floating-point literal') from exc

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f'{x} not in range [0.0, 1.0]')

    return x


def _path_exists(path, parser):
    """Ensure a given path exists."""
    if path is None or not Path(path).exists():
        raise parser.error(f'Path does not exist: <{path}>.')
    return Path(path).absolute()


def _is_file(path, parser):
    """Ensure a given path exists and it is a file."""
    path = _path_exists(path, parser)
    if not path.is_file():
        raise parser.error(f'Path should point to a file (or symlink of file): <{path}>.')
    return path


def _process_value(value):
    import bids

    if value is None:
        return bids.layout.Query.NONE
    elif value == '*':
        return bids.layout.Query.ANY
    else:
        return value


def _filter_pybids_none_any(dct):
    d = {}
    for k, v in dct.items():
        if isinstance(v, list):
            d[k] = [_process_value(val) for val in v]
        else:
            d[k] = _process_value(v)
    return d


def _bids_filter(value, parser):
    from json import JSONDecodeError, loads

    if value:
        if Path(value).exists():
            try:
                return loads(Path(value).read_text(), object_hook=_filter_pybids_none_any)
            except JSONDecodeError as exc:
                raise parser.error(f'JSON syntax error in: <{value}>.') from exc
        else:
            raise parser.error(f'Path does not exist: <{value}>.')


def _min_one(value, parser):
    """Ensure an argument is not lower than 1."""
    value = int(value)
    if value < 1:
        raise parser.error("Argument can't be less than one.")
    return value


class YesNoAction(Action):
    """A custom argparse "store" action to handle "y", "n", None, "auto" values."""

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: U100
        """Call the argument."""
        lookup = {'y': True, 'n': False, None: True, 'auto': 'auto'}
        if values not in lookup:
            raise parser.error(f"Invalid value '{values}' for {self.dest}")

        setattr(namespace, self.dest, lookup[values])


class ToDict(Action):
    """A custom argparse "store" action to handle a list of key=value pairs."""

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: U100
        """Call the argument."""
        d = {}
        for spec in values:
            try:
                name, loc = spec.split('=')
                loc = Path(loc)
            except ValueError:
                loc = Path(spec)
                name = loc.name

            if name in d:
                raise parser.error(f'Received duplicate derivative name: {name}')
            elif name == 'preprocessed':
                raise parser.error("The 'preprocessed' derivative is reserved for internal use.")

            d[name] = loc
        setattr(namespace, self.dest, d)


class ConfoundsAction(Action):
    """A custom argparse "store" action to handle a path or ."""

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: U100
        """Call the argument."""
        builtins = [
            'auto',
            '27P',
            '36P',
            '24P',
            'acompcor',
            'aroma',
            'acompcor_gsr',
            'aroma_gsr',
            'none',
            'gsr_only',
        ]
        if values in builtins:
            setattr(namespace, self.dest, values)
        else:
            if not Path(values).exists():
                raise parser.error(f'Nuisance configuration does not exist: <{values}>.')

            setattr(namespace, self.dest, Path(values))
