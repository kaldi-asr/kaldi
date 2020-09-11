"""Functions for reading/writing and manipulating NIST un-partitioned
evaluation maps.

An un-partitioned evaluation map (UEM) specifies the time regions within each
file that will be scored.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict
try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping
import itertools
import os

from intervaltree import IntervalTree

from .six import iterkeys
from .utils import format_float

__all__ = ['gen_uem', 'load_uem', 'write_uem', 'UEM']


class UEM(MutableMapping):
    """Un-partitioned evaluaion map (UEM).

    A UEM defines a mapping from file ids to scoring regions.
    """
    def __init__(self, *args, **kwargs):
        super(UEM, self).__init__()
        self.update(*args, **kwargs)

    def __setitem__(self, fid, score_regions):
        # Validate types. Expects sequence of (onset, offset) pairs.
        invalid_type_msg = (
            'Expected sequence of pairs. Received: %r (%s).' %
            (score_regions, type(score_regions)))
        try:
            score_regions = [tuple(region) for region in score_regions]
        except TypeError:
            raise TypeError(invalid_type_msg)
        for score_region in score_regions:
            if len(score_region) != 2:
                raise TypeError(invalid_type_msg)

        # Validate that the (onset, offset) pairs are valid: no negative
        # timestamps or negative durations.
        def _convert_to_float(score_region):
            onset, offset = score_region
            try:
                onset = float(onset)
                offset = float(offset)
            except ValueError:
                raise ValueError(
                    'Could not convert interval onset/offset to float: %s' %
                    repr(score_region))
            if onset >= offset or onset < 0:
                raise ValueError(
                    'Invalid interval (%.3f, %.3f) for file "%s".' %
                    (onset, offset, fid))
            return onset, offset
        score_regions = [_convert_to_float(region) for region in score_regions]

        # Merge overlaps. Use of intervaltree Incurs some additional overhead,
        # but pretty compared to actual scoring.
        tree = IntervalTree.from_tuples(score_regions)
        tree.merge_overlaps()
        score_regions = [(intrvl.begin, intrvl.end) for intrvl in tree]

        self.__dict__[fid] = score_regions

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return '{}, UEM({})'.format(super(UEM, self).__repr__(), self.__dict__)


def load_uem(uemf):
    """Load un-partitioned evaluation map from file in NIST format.

    The un-partitioned evaluation map (UEM) file format contains
    one record per line, each line consisting of NN space-delimited
    fields:

    - file id  --  file id
    - channel  --  channel (1-indexed)
    - onset  --  onset of evaluation region in seconds from beginning of file
    - offset  --  offset of evaluation region in seconds from beginning of
      file

    Lines beginning with semicolons are regarded as comments and ignored.

    Parameters
    ----------
    uemf : str
        Path to UEM file.

    Returns
    -------
    uem : UEM
        Evaluation map.
    """
    with open(uemf, 'rb') as f:
        fid_to_score_regions = defaultdict(list)
        for line in f:
            if line.startswith(b';'):
                continue
            fields = line.decode('utf-8').strip().split()
            file_id = os.path.splitext(fields[0])[0]
            onset = float(fields[2])
            offset = float(fields[3])
            fid_to_score_regions[file_id].append((onset, offset))
    return UEM(fid_to_score_regions.items())


def write_uem(uemf, uem, n_digits=3):
    """Write un-partitioned evaluation map to file in NIST format.

    Parameters
    ----------
    uemf : str
        Path to output UEM file.

    uem : UEM
        Evaluation map.

    n_digits : int, optional
        Number of decimal digits to round to.
        (Default: 3)
    """
    with open(uemf, 'wb') as f:
        for file_id in sorted(iterkeys(uem)):
            for onset, offset in sorted(uem[file_id]):
                line = ' '.join([file_id,
                                 '1',
                                 format_float(onset, n_digits),
                                 format_float(offset, n_digits)
                                ])
                f.write(line.encode('utf-8'))
                f.write(b'\n')


def gen_uem(ref_turns, sys_turns):
    """Generate un-partitioned evaluation map.

    For each file, the extent of the scoring region is set as follows:

    - onset = min(minimum reference onset, minimum system onset)
    - offset = max(maximum reference onset, maximum system offset)

    Parameters
    ----------
    ref_turns : list of Turn
        Reference speaker turns.

    sys_turns : list of Turn
        System speaker turns.

    Returns
    -------
    uem : UEM
        Un-partitioned evaluation map.
    """
    file_ids = set()
    onsets = defaultdict(set)
    offsets = defaultdict(set)
    for turn in itertools.chain(ref_turns, sys_turns):
        file_ids.add(turn.file_id)
        onsets[turn.file_id].add(turn.onset)
        offsets[turn.file_id].add(turn.offset)
    uem = UEM()
    for file_id in file_ids:
        uem[file_id] = [(min(onsets[file_id]), max(offsets[file_id]))]
    return uem
