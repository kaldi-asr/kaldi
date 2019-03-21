#!/usr/bin/env python3

# Copyright 2018  Vimal Manohar
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import collections
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description="""
        This script combines segments into utterances at
        recording-level and write out new utt2spk file with reco-id as the
        speakers. If --write-reco2utt is provided, it writes a mapping from
        recording-id to the list of utterances sorted by start and end times.
        This map can be used to combine text corresponding to the segments to
        recording-level.""")

    parser.add_argument("--write-reco2utt", help="If provided, writes a "
                        "mapping from recording-id to list of utterances "
                        "sorted by start and end times.")
    parser.add_argument("segments_in", help="Input segments file")
    parser.add_argument("utt2spk_out", help="Output utt2spk file")

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    utt2reco = {}
    segments_for_reco = defaultdict(list)
    for line in open(args.segments_in):
        parts = line.strip().split()

        if len(parts) < 4:
            raise TypeError("bad line in segments file {}".format(line))

        utt = parts[0]
        reco = parts[1]
        start_time = parts[2]
        end_time = parts[3]

        segments_for_reco[reco].append((utt, start_time, end_time))
        utt2reco[utt] = reco

    if args.write_reco2utt is not None:
        with open(args.write_reco2utt, 'w') as reco2utt_writer, \
                open(args.utt2spk_out, 'w') as utt2spk_writer:
            for reco, segments_in_reco in segments_for_reco.items():
                utts = ' '.join([seg[0] for seg in sorted(
                    segments_in_reco, key=lambda x:(x[1], x[2]))])
                print("{0} {1}".format(reco, utts), file=reco2utt_writer)
                print ("{0} {0}".format(reco), file=utt2spk_writer)
    else:
        with open(args.utt2spk_out, 'w') as utt2spk_writer:
            for reco in segments_for_reco.keys():
                print ("{0} {0}".format(reco), file=utt2spk_writer)


if __name__ == "__main__":
    main()
