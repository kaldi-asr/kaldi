#!/usr/bin/env python3

# Copyright     2017  Hossein Hadian
# Apache 2.0


""" This script finds a set of allowed lengths for a given OCR/HWR data dir.
    The allowed lengths are spaced by a factor (like 10%) and are written
    in an output file named "allowed_lengths.txt" in the output data dir. This
    file is later used by make_features.py to pad each image sufficiently so that
    they all have an allowed length. This is intended for end2end chain training.
"""
from __future__ import division

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
    parser = argparse.ArgumentParser(description="""This script finds a set of
                                   allowed lengths for a given OCR/HWR data dir.
                                   Intended for chain training.""")
    parser.add_argument('factor', type=float, default=12,
                        help='Spacing (in percentage) between allowed lengths.')
    parser.add_argument('srcdir', type=str,
                        help='path to source data dir')
    parser.add_argument('--coverage-factor', type=float, default=0.05,
                        help="""Percentage of durations not covered from each
                             side of duration histogram.""")
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
            line = line.strip()
            sp_pos = line.find(' ')
            key = line[:sp_pos]
            val = line[sp_pos+1:]
            m[key] = val
    return m

def find_duration_range(img2len, coverage_factor):
    """Given a list of utterances, find the start and end duration to cover

     If we try to cover
     all durations which occur in the training set, the number of
     allowed lengths could become very large.

     Returns
     -------
     start_dur: int
     end_dur: int
    """
    durs = []
    for im, imlen in img2len.items():
        durs.append(int(imlen))
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
    if start_dur < 30:
        start_dur = 30  # a hard limit to avoid too many allowed lengths --not critical
    return start_dur, end_dur


def find_allowed_durations(start_len, end_len, args):
    """Given the start and end duration, find a set of
       allowed durations spaced by args.factor%. Also write
       out the list of allowed durations and the corresponding
       allowed lengths (in frames) on disk.

     Returns
     -------
     allowed_durations: list of allowed durations (in seconds)
    """

    allowed_lengths = []
    length = start_len
    with open(os.path.join(args.srcdir, 'allowed_lengths.txt'), 'w', encoding='latin-1') as fp:
        while length < end_len:
            if length % args.frame_subsampling_factor != 0:
                length = (args.frame_subsampling_factor *
                          (length // args.frame_subsampling_factor))
            allowed_lengths.append(length)
            fp.write("{}\n".format(int(length)))
            length = max(length * args.factor, length + args.frame_subsampling_factor)
    return allowed_lengths



def main():
    args = get_args()
    args.factor = 1.0 + args.factor/100.0

    image2length = read_kaldi_mapfile(os.path.join(args.srcdir, 'image2num_frames'))

    start_dur, end_dur = find_duration_range(image2length, args.coverage_factor)
    logger.info("Lengths in the range [{},{}] will be covered. "
                "Coverage rate: {}%".format(start_dur, end_dur,
                                      100.0 - args.coverage_factor * 2))
    logger.info("There will be {} unique allowed lengths "
                "for the images.".format(int((math.log(float(end_dur)/start_dur))/
                                             math.log(args.factor))))

    allowed_durations = find_allowed_durations(start_dur, end_dur, args)


if __name__ == '__main__':
      main()
