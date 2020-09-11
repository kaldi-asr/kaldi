"""Custom argument parser and action classes."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import sys


__all__ = ['ArgumentParser']


class ArgumentParser(argparse.ArgumentParser):
    """Sub-class of ``ArgumentParser`` that write errors to STDERR."""
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
