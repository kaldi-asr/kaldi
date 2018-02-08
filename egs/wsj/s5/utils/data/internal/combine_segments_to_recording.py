#!/usr/bin/env python

from __future__ import print_function
import argparse
import sys
import collections
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description="""
        This script combines segments into utterances at
        recording-level and write out new utt2spk file with reco-id as the
        speakers. If text-in and text-out are provided, then
        the transcription from text-in are combined into recording-level
        transcription and written to text-out.""")

    parser.add_argument("--text-in", help="Input text file")
    parser.add_argument("--text-out", help="Output text file")
    parser.add_argument("segments_in", help="Input segments file")
    parser.add_argument("utt2spk_out", help="Output utt2spk file")

    args = parser.parse_args()

    if args.text_in is not None:
        if args.text_out is None:
            raise Exception("--text-out is required if --text-in is provided.")

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

    text = {}
    if args.text_in is not None:
        for line in open(args.text_in):
            parts = line.strip().split()
            text[parts[0]] = " ".join(parts[1:])

        with open(args.text_out, 'w') as text_writer, \
                open(args.utt2spk_out, 'w') as utt2spk_writer:
            for reco, segments_in_reco in segments_for_reco.items():
                text_for_reco = " ".join([text[seg[0]] for seg in sorted(
                    segments_in_reco, key=lambda x:(x[1], x[2]))])
                print("{0} {1}".format(reco, text_for_reco), file=text_writer)
                print ("{0} {0}".format(reco), file=utt2spk_writer)
    else:
        with open(args.utt2spk_out, 'w') as utt2spk_writer:
            for reco in segments_for_reco.keys():
                print ("{0} {0}".format(reco), file=utt2spk_writer)


if __name__ == "__main__":
    main()
