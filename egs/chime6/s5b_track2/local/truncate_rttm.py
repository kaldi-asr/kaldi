#!/usr/bin/env python
# Apache 2.0
#
# This script truncates the rttm file using UEM file and writes it to a new rttm file.
# We use some utility functions from the dscore toolkit.

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
from dscore.scorelib.turn import Turn, trim_turns
from dscore.scorelib.uem import load_uem
from dscore.scorelib.utils import format_float

sys.path.insert(0, 'steps')
import libs.common as common_lib


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script truncates the rttm file
                       using UEM file""")
    parser.add_argument("rttm_file", type=str,
                        help="""Input RTTM file.
                        The format of the RTTM file is
                        <type> <file-id> <channel-id> <begin-time> """
                        """<end-time> <NA> <NA> <speaker> <conf>""")
    parser.add_argument("uem_file", type=str,
                        help="""Input UEM file.
                        The format of the UEM file is
                        <file-id> <channel-id> <begin-time> <end-time>""")
    parser.add_argument("rttm_file_write", type=str,
                        help="""output RTTM file.""")
    parser.add_argument("--min-segment-length", type=float,
                        help="""Minimum segment length to keep""")
    args = parser.parse_args()
    return args

# The following load_rttm and write_rttm are modified from dscore to also
# work with pipe input and output.

def _parse_rttm_line(line):
    line = line.strip()
    fields = line.split()
    if len(fields) < 9:
        raise IOError('Number of fields < 9. LINE: "%s"' % line)
    file_id = fields[1]
    speaker_id = fields[7]

    # Check valid turn onset.
    try:
        onset = float(fields[3])
    except ValueError:
        raise IOError('Turn onset not FLOAT. LINE: "%s"' % line)
    if onset < 0:
        raise IOError('Turn onset < 0 seconds. LINE: "%s"' % line)

    # Check valid turn duration.
    try:
        dur = float(fields[4])
    except ValueError:
        raise IOError('Turn duration not FLOAT. LINE: "%s"' % line)
    if dur <= 0:
        raise IOError('Turn duration <= 0 seconds. LINE: "%s"' % line)

    return Turn(onset, dur=dur, speaker_id=speaker_id, file_id=file_id)

def load_rttm(rttmf):
    with common_lib.smart_open(rttmf, 'r') as f:
        turns = []
        speaker_ids = set()
        file_ids = set()
        for line in f:
            if line.startswith('SPKR-INFO'):
                continue
            turn = _parse_rttm_line(line)
            turns.append(turn)
            speaker_ids.add(turn.speaker_id)
            file_ids.add(turn.file_id)
    return turns, speaker_ids, file_ids

def write_rttm(rttmf, turns, n_digits=3):
    with common_lib.smart_open(rttmf, 'w') as f:
        for turn in sorted(turns, key=lambda x:x.onset):
            fields = ['SPEAKER',
                      turn.file_id,
                      '1',
                      format_float(turn.onset, n_digits),
                      format_float(turn.dur, n_digits),
                      '<NA>',
                      '<NA>',
                      turn.speaker_id,
                      '<NA>',
                      '<NA>']
            line = ' '.join(fields)
            f.write(line + '\n')

if __name__ == '__main__':
    args = get_args()
    turns, speaker_ids, file_ids = load_rttm(args.rttm_file)
    loaded_uem = load_uem(args.uem_file)
    truncated_turns = trim_turns(turns, loaded_uem)
    truncated_turns = [turn for turn in truncated_turns 
                       if turn.dur >= args.min_segment_length]
    write_rttm(args.rttm_file_write, truncated_turns)
