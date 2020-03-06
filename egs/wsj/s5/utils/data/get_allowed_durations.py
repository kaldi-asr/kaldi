#!/usr/bin/env python3

# Copyright     2017  Hossein Hadian
#               2019  Facebook Inc. (Author: Vimal Manohar)
# Apache 2.0


""" This script generates a set of allowed lengths of utterances
    spaced by a factor (like 10%). This is useful for generating
    fixed-length chunks for chain training.
"""

import argparse
import os
import sys
import copy
import math
import logging

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_args():
    parser = argparse.ArgumentParser(description="""
    This script creates a list of allowed durations of utterances for flatstart
    LF-MMI training corresponding to input data directory 'data_dir' and writes
    it in two files in output directory 'dir':
    1) allowed_durs.txt -- durations are in seconds
    2) allowed_lengths.txt -- lengths are in number of frames

    Both the allowed_durs.txt and allowed_lengths.txt are formatted to
    have one entry on each line. Examples are as follows:

    $ echo data/train/allowed_lengths.txt
    414
    435
    468

    $ echo data/train/allowed_durs.txt
    4.16
    4.37
    4.70

    These files can then be used by a downstream script to perturb the
    utterances to these lengths.
    A perturbed data directory (created by a downstream script
    similar to utils/data/perturb_speed_to_allowed_lengths.py)
    that only contains utterances of these allowed durations,
    along with the corresponding allowed_lengths.txt are
    consumed by the e2e chain egs preparation script.
    See steps/nnet3/chain/e2e/get_egs_e2e.sh for how these are used.

    See also:
    * egs/cifar/v1/image/get_allowed_lengths.py -- a similar script for OCR datasets
    * utils/data/perturb_speed_to_allowed_lengths.py --
        creates the allowed_lengths.txt AND perturbs the data directory
    """)
    parser.add_argument('factor', type=float, default=12,
                        help='Spacing (in percentage) between allowed lengths. '
                        'Can be 0, which means all seen lengths that are a multiple of '
                        'frame_subsampling_factor will be allowed.')
    parser.add_argument('data_dir', type=str, help='path to data dir. Assumes that '
                        'it contains the utt2dur file.')
    parser.add_argument('dir', type=str, help='We write the output files '
                        'allowed_lengths.txt and allowed_durs.txt to this directory.')
    parser.add_argument('--coverage-factor', type=float, default=0.05,
                        help="""Percentage of durations not covered from each
                             side of duration histogram.""")
    parser.add_argument('--frame-shift', type=int, default=10,
                        help="""Frame shift in milliseconds.""")
    parser.add_argument('--frame-length', type=int, default=25,
                        help="""Frame length in milliseconds.""")
    parser.add_argument('--frame-subsampling-factor', type=int, default=3,
                        help="""Chain frame subsampling factor.
                             See steps/nnet3/chain/train.py""")
    args = parser.parse_args()
    return args


def read_kaldi_mapfile(path):
    """ Read any Kaldi mapping file - like text, .scp files, etc.
    """

    m = {}
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip(" \t\r\n")
            sp_pos = line.find(' ')
            key = line[:sp_pos]
            val = line[sp_pos+1:]
            m[key] = val
    return m


def find_duration_range(utt2dur, coverage_factor):
    """Given a list of utterance durations, find the start and end duration to cover

     If we try to cover
     all durations which occur in the training set, the number of
     allowed lengths could become very large.

     Returns
     -------
     start_dur: float
     end_dur: float
    """
    durs = [float(val) for key, val in utt2dur.items()]
    durs.sort()
    to_ignore_dur = 0
    tot_dur = sum(durs)
    for d in durs:
        to_ignore_dur += d
        if to_ignore_dur * 100.0 / tot_dur > coverage_factor:
            start_dur = d
            break
    to_ignore_dur = 0
    for d in reversed(durs):
        to_ignore_dur += d
        if to_ignore_dur * 100.0 / tot_dur > coverage_factor:
            end_dur = d
            break
    if start_dur < 0.3:
        start_dur = 0.3  # a hard limit to avoid too many allowed lengths --not critical
    return start_dur, end_dur


def get_allowed_durations(start_dur, end_dur, args):
    """Given the start and end duration, find a set of
       allowed durations spaced by args.factor%. Also write
       out the list of allowed durations and the corresponding
       allowed lengths (in frames) on disk.

     Returns
     -------
     allowed_durations: list of allowed durations (in seconds)
    """

    allowed_durations = []
    d = start_dur
    with open(os.path.join(args.dir, 'allowed_durs.txt'), 'w', encoding='latin-1') as durs_fp, \
           open(os.path.join(args.dir, 'allowed_lengths.txt'), 'w', encoding='latin-1') as lengths_fp:
        while d < end_dur:
            length = int(d * 1000 - args.frame_length) / args.frame_shift + 1
            if length % args.frame_subsampling_factor != 0:
                length = (args.frame_subsampling_factor *
                              (length // args.frame_subsampling_factor))
                d = (args.frame_shift * (length - 1.0)
                     + args.frame_length + args.frame_shift / 2) / 1000.0
            allowed_durations.append(d)
            durs_fp.write("{}\n".format(d))
            lengths_fp.write("{}\n".format(int(length)))
            d *= args.factor
    return allowed_durations


def get_trivial_allowed_durations(utt2dur, args):
    lengths = list(set(
        [int(float(d) * 1000 - args.frame_length) / args.frame_shift + 1
         for key, d in utt2dur.items()]
    ))
    lengths.sort()

    allowed_durations = []
    with open(os.path.join(args.dir, 'allowed_durs.txt'), 'w', encoding='latin-1') as durs_fp, \
           open(os.path.join(args.dir, 'allowed_lengths.txt'), 'w', encoding='latin-1') as lengths_fp:
        for length in lengths:
            if length % args.frame_subsampling_factor != 0:
                length = (args.frame_subsampling_factor *
                              (length // args.frame_subsampling_factor))
                d = (args.frame_shift * (length - 1.0)
                     + args.frame_length + args.frame_shift / 2) / 1000.0
            allowed_durations.append(d)
            durs_fp.write("{}\n".format(d))
            lengths_fp.write("{}\n".format(int(length)))

    assert len(allowed_durations) > 0
    start_dur = allowed_durations[0]
    end_dur = allowed_durations[-1]

    logger.info("Durations in the range [{},{}] will be covered."
                "".format(start_dur, end_dur))
    logger.info("There will be {} unique allowed lengths "
                "for the utterances.".format(len(allowed_durations)))

    return allowed_durations


def main():
    args = get_args()
    utt2dur = read_kaldi_mapfile(os.path.join(args.data_dir, 'utt2dur'))

    if args.factor == 0.0:
        get_trivial_allowed_durations(utt2dur, args)
        return

    args.factor = 1.0 + args.factor / 100.0

    start_dur, end_dur = find_duration_range(utt2dur, args.coverage_factor)
    logger.info("Durations in the range [{},{}] will be covered. "
                "Coverage rate: {}%".format(start_dur, end_dur,
                                      100.0 - args.coverage_factor * 2))
    logger.info("There will be {} unique allowed lengths "
                "for the utterances.".format(int(math.log(end_dur / start_dur)/
                                                 math.log(args.factor))))

    get_allowed_durations(start_dur, end_dur, args)


if __name__ == '__main__':
      main()
