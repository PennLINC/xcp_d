"""Manipulate Python warnings."""

import logging
import warnings

_wlog = logging.getLogger('py.warnings')
_wlog.addHandler(logging.NullHandler())


def _warn(message, category=None, stacklevel=1, source=None, **kwargs):
    """Redefine the warning function."""
    if category is not None:
        category = type(category).__name__
        category = category.replace('type', 'WARNING')

    if kwargs:
        logging.getLogger('py.warnings').warning(f'Extra warning kwargs: {kwargs}')

    logging.getLogger('py.warnings').warning(f'{category or "WARNING"}: {message}')


def _showwarning(message, category, filename, lineno, file=None, line=None, **kwargs):
    _warn(message, category=category, **kwargs)


warnings.warn = _warn
warnings.showwarning = _showwarning
