#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse


def sort_utts_per_conversation(in_f, out_f):
    r"""Sort utterances per conversation by the starting time of each utterance.

    Args:
        in_f:   Input file contains utterances of two-party conversations.
                An example format can be:
                    en_4156-A_030185-030248 OH YEAH
                    en_4156-A_030470-030672 WELL I AM GOING TO HAVE MINE IN TWO MORE CLASSES
                    en_4156-A_030763-031116 NO I AM NOT WELL THEN I HAVE TO TAKE MY EXAMS MY ORALS BUT
                    ...
                    en_4156-B_029874-030166 EVERYBODY HAS A MASTER IS OUT HERE
                    en_4156-B_030297-030472 WELL IT SEEMS LIKE IT
                    en_4156-B_030683-030795 YOU ARE KIDDING
                    ...
                For utterance id 'en_4156-A_030185-030248', 'en' means English,
                '4156' is the conversation id, 'A' and 'B' are two speakers,
                '030185' is the starting time, and '030248' is the ending time.
        out_f:  Output file with utterances sorted by their starting time.
                For the given example of input, the output is:
                    EVERYBODY HAS A MASTER IS OUT HERE
                    OH YEAH
                    WELL IT SEEMS LIKE IT
                    WELL I AM GOING TO HAVE MINE IN TWO MORE CLASSES
                    YOU ARE KIDDING
                    NO I AM NOT WELL THEN I HAVE TO TAKE MY EXAMS MY ORALS BUT
                    ...
    """

    utts = []
    with open(in_f, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            utts.append({'utt_id': line[0:7], 'speaker': line[7:10],
                         'start_time': line[10:16], 'end_time': line[16:24],
                         'utt': line[24:]})

    utts_sorted = sorted(utts, key=lambda x: (x['utt_id'], x['start_time']))

    with open(out_f, 'w', encoding='utf-8') as f:
        for line in utts_sorted:
            f.write(line['utt_id'] + line['speaker'] + line['start_time'] +
                    line['end_time'] + line['utt'])


def main():
    parser = argparse.ArgumentParser(description="Sort utterances per "
                                     "conversation of SWBD dataset by the "
                                     "starting time of each utterance.")
    parser.add_argument('--infile', type=str, required=True,
                        help="File to be sorted.")
    parser.add_argument('--outfile', type=str, required=True,
                        help="Sorted output file.")
    args = parser.parse_args()
    assert os.path.exists(args.infile), "Text file does not exists."

    sort_utts_per_conversation(args.infile, args.outfile)


if __name__ == '__main__':
    main()
