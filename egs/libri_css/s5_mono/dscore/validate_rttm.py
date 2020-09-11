#!/usr/bin/env python
"""Validate RTTM files.

To validate a RTTM files ``f1.rttm``, ``f2.rttm``, ...

    python validate_rttm.py f1.rttm f2.rttm ...

which will for each file report the following:

- the number of unique file ids found
- the number of speaker ids found
- each line containing an error + an error message
"""
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import sys

from scorelib import __version__ as VERSION
from scorelib.rttm import validate_rttm
from scorelib.utils import error, info


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description='Validate RTTM files.', add_help=True,
        usage='%(prog)s [options] rttm_fns')
    parser.add_argument(
        'rttm_fns', nargs='+', help='RTTM files')
    parser.add_argument(
        '--version', action='version',
        version='%(prog)s ' + VERSION)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    for rttm_fn in args.rttm_fns:
        info('Validating %s...' % rttm_fn)
        file_ids, speaker_ids, error_messages = validate_rttm(rttm_fn)
        file_ids = sorted(file_ids)
        info('%d file ids found: %s' %
             (len(file_ids), ', '.join(file_ids)))
        speaker_ids = sorted(speaker_ids)
        info('%d speaker ids found: %s' %
             (len(speaker_ids), ', '.join(speaker_ids)))
        for msg in error_messages:
            error(msg, file=sys.stdout)
