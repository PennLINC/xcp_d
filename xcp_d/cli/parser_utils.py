"""Utility functions for xcp_d command-line interfaces."""
import argparse
import json
import logging
import os
import warnings
from argparse import Action

from niworkflows import NIWORKFLOWS_LOG

warnings.filterwarnings("ignore")

logging.addLevelName(25, "IMPORTANT")  # Add a new level between INFO and WARNING
logging.addLevelName(15, "VERBOSE")  # Add a new level between INFO and DEBUG
logger = logging.getLogger("cli")


def json_file(file_):
    """Load a JSON file and return it."""
    if file_ is None:
        return file_
    elif os.path.isfile(file_):
        with open(file_, "r") as fo:
            data = json.load(fo)
        return data
    else:
        raise ValueError(f"Not supported: {file_}")


def _warn_redirect(message, category):
    logger.warning("Captured warning (%s): %s", category, message)


def check_deps(workflow):
    """Check the dependencies for the workflow."""
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, "_cmd") and which(node.interface._cmd.split()[0]) is None)
    )


def _int_or_auto(string, is_parser=True):
    """Check if argument is an integer >= 0 or the string "auto"."""
    if string == "auto":
        return string

    error = argparse.ArgumentTypeError if is_parser else ValueError
    try:
        intarg = int(string)
    except ValueError:
        msg = "Argument must be a nonnegative integer or 'auto'."
        raise error(msg)

    if intarg < 0:
        raise error("Int argument must be nonnegative.")
    return intarg


def _float_or_auto(string, is_parser=True):
    """Check if argument is a float >= 0 or the string "auto"."""
    if string == "auto":
        return string

    error = argparse.ArgumentTypeError if is_parser else ValueError
    try:
        floatarg = float(string)
    except ValueError:
        msg = "Argument must be a nonnegative float or 'auto'."
        raise error(msg)

    if floatarg < 0:
        raise error("Float argument must be nonnegative.")
    return floatarg


def _restricted_float(x):
    """From https://stackoverflow.com/a/12117065/2589328."""
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")

    return x


class _DeprecatedStoreAction(Action):
    """A custom argparse "store" action to raise a DeprecationWarning.

    Based off of https://gist.github.com/bsolomon1124/44f77ed2f15062c614ef6e102bc683a5.
    """

    __version__ = ""

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: U100
        """Call the argument."""
        NIWORKFLOWS_LOG.warn(
            f"Argument '{option_string}' is deprecated and will be removed in version "
            f"{self.__version__}. "
        )
        setattr(namespace, self.dest, values)


class _DeprecatedStoreAction040(_DeprecatedStoreAction):
    __version__ = "0.4.0"
