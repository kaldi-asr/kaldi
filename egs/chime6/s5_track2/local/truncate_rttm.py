#!/usr/bin/env python3
# Apache 2.0
# This script truncates the rttm file
# using UEM file and writes it to a new rttm file
#
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from scorelib.turn import trim_turns
import scorelib.rttm as rttm_func
from scorelib.uem import load_uem

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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    rttm_writer = open(args.rttm_file_write, 'w')
    turns, speaker_ids, file_ids = rttm_func.load_rttm(args.rttm_file)
    loaded_uem = load_uem(args.uem_file)
    truncated_turns = trim_turns(turns, loaded_uem)
    rttm_func.write_rttm(args.rttm_file_write,truncated_turns)
