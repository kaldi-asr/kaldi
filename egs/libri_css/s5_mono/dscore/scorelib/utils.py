"""Utility functions."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import itertools
import sys

from . import six

__all__ = ['clip', 'error', 'format_float', 'groupby', 'info', 'warn', 'xor']


def error(msg, file=sys.stderr):
    """Log error message ``msg`` to stderr."""
    msg = 'ERROR: %s' % msg
    if six.PY2:
        msg = msg.encode('utf-8')
    print(msg, file=file)


def info(msg, print_level=False, file=sys.stdout):
    """Log info message ``msg`` to stdout."""
    if print_level:
        msg = 'INFO: %s' %msg
    if six.PY2:
        msg = msg.encode('utf-8')
    print(msg, file=file)


def warn(msg, file=sys.stderr):
    """Log warning message ``msg`` to stderr."""
    msg = 'WARNING: %s' %msg
    if six.PY2:
        msg = msg.encode('utf-8')
    print(msg, file=file)


def xor(x, y):
    """Return truth value of ``x`` XOR ``y``."""
    return bool(x) != bool(y)


def format_float(x, n_digits=3):
    """Format floating point number for output as string.

    Parameters
    ----------
    x : float
        Number.

    n_digits : int, optional
        Number of decimal digits to round to.
        (Default: 3)

    Returns
    -------
    s : str
        Formatted string.
    """
    fmt_str = '%%.%df' % n_digits
    return fmt_str % round(x, n_digits)


def clip(x, lower, upper):
    """Clip ``x`` to [``lower``, ``upper``]."""
    return min(max(x, lower), upper)


def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group
