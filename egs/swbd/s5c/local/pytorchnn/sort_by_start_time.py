#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse


def sort_utts_per_conversation(infile, outfile):
    utts = []
    with open(infile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            utts.append({'utt_id':line[0:7],'speaker':line[7:10],'start_time':line[10:16],'end_time':line[16:24],'utt':line[24:]})

    utts_sorted = sorted(utts, key = lambda x:(x['utt_id'], x['start_time']))

    with open(outfile, 'w', encoding='utf-8') as f:
        for line in utts_sorted:
            f.write(line['utt_id'] + line['speaker'] + line['start_time'] + line['end_time'] + line['utt'])


def main():
    parser = argparse.ArgumentParser(description="Sort utterances per conversation of SWBD dataset by utterances' starting time.")
    parser.add_argument('--infile', type=str, required=True,
                        help="File to be sorted.")
    parser.add_argument('--outfile', type=str, required=True,
                        help="Sorted output file.")
    args = parser.parse_args()
    assert os.path.exists(args.infile), "Text file does not exists."

    sort_utts_per_conversation(args.infile, args.outfile)


if __name__ == '__main__':
    main()
