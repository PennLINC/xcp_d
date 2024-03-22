"""Manipulate Python warnings."""
import logging
import warnings

_wlog = logging.getLogger("py.warnings")
_wlog.addHandler(logging.NullHandler())


def _warn(message, category=None, stacklevel=1, source=None):
    """Redefine the warning function."""
    if category is not None:
        category = type(category).__name__
        category = category.replace("type", "WARNING")

    logging.getLogger("py.warnings").warning(f"{category or 'WARNING'}: {message}")


def _showwarning(message, category, filename, lineno, file=None, line=None):
    _warn(message, category=category)


warnings.warn = _warn
warnings.showwarning = _showwarning
