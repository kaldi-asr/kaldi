#!/usr/bin/env python3
# Apache 2.0
# This script truncates the rttm file
# using UEM file and writes it to a new rttm file
#
from __future__ import print_function
from __future__ import unicode_literals
from scorelib.uem import UEM

import argparse
from scorelib.turn import trim_turns
import scorelib.rttm as rttm_func

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script truncates the rttm file
                       using UEM file""")
    parser.add_argument("rttm_file", type=str,
                        help="""Input RTTM file.
                            The format of the RTTM file is
                            <type> <file-id> <channel-id> <begin-time> """
                             """<end-time> <NA> <NA> <speaker> <conf>""")
    parser.add_argument("rttm_file_write", type=str,
                        help="""output RTTM file.""")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    rttm_writer = open(args.rttm_file_write, 'w')
    turns, speaker_ids, file_ids = rttm_func.load_rttm(args.rttm_file)
    uem = UEM({
        'S01_U01': [(0, 12000)],
        'S02_U01': [(75, 12000)],
        'S09_U01': [(64, 12000)],
        'S21_U01': [(59, 12000)],
        'S01_U02': [(0, 12000)],
        'S02_U02': [(75, 12000)],
        'S09_U02': [(64, 12000)],
        'S21_U02': [(59, 12000)],
        'S01_U03': [(0, 12000)],
        'S02_U03': [(75, 12000)],
        'S09_U03': [(64, 12000)],
        'S21_U03': [(59, 12000)],
        'S01_U04': [(0, 12000)],
        'S02_U04': [(75, 12000)],
        'S09_U04': [(64, 12000)],
        'S21_U04': [(59, 12000)],
        'S01_U06': [(0, 12000)],
        'S02_U06': [(75, 12000)],
        'S09_U06': [(64, 12000)],
        'S21_U06': [(59, 12000)]})
    truncated_turns = trim_turns(turns, uem)
    rttm_func.write_rttm(args.rttm_file_write,truncated_turns)
