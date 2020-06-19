#! /usr/bin/env python

# Copyright 2017  Vimal Manohar
#           2017  Matthew Maciejewski
#           2019  Yiming Wang
# Apache 2.0.

from __future__ import print_function
import argparse
import logging
import random
import sys
import textwrap

def get_args():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
        Creates a subsegments file from an input segments file
        that has the format
        <subsegment-id> <utterance-id> <start-time> <end-time>,
        where the timing are relative to the start-time of the
        <utterance-id> in the input segments file.

        e.g.: get_uniform_subsegments.py data/dev/segments > \\
                data/dev_uniform_segments/sub_segments

        utils/data/subsegment_data_dir.sh data/dev \\
            data/dev_uniform_segments/sub_segments data/dev_uniform_segments

        The output is written to stdout. The resulting file can be
        passed to utils/data/subsegment_data_dir.sh to sub-segment
        the data directory."""),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overlap-duration", type=float,
                        default=0.3, help="""Overlap between
                        adjacent segments (in seconds)""")
    parser.add_argument("--max-remaining-duration", type=float,
                        default=0.3, help="""Segment is not split
                        if the left-over duration is more than this
                        many seconds""")
    parser.add_argument("--seed", type=int,
                        default=0, help="""random seed""")
    parser.add_argument("segments_file", type=argparse.FileType('r'),
                        help="""Input kaldi segments file""")
    parser.add_argument("positive_duration_file", type=argparse.FileType('r'),
                        help="""the file containing utt2dur entries for positive
                        examples""")

    args = parser.parse_args()
    return args


def run(args):
    positive_dur = [float(line.strip().split()[1]) for line in args.positive_duration_file]
    num_pos_utts = len(positive_dur)
    random.seed(args.seed)
    for line in args.segments_file:
        parts = line.strip().split()
        utt_id = parts[0]
        start_time = float(parts[2])
        end_time = float(parts[3])

        dur = end_time - start_time

        start = start_time
        this_dur = positive_dur[random.randrange(num_pos_utts)]
        while (dur > this_dur + args.max_remaining_duration):
            end = start + this_dur
            start_relative = start - start_time
            end_relative = end - start_time
            new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
                utt_id=utt_id, s=int(100 * start_relative),
                e=int(100 * end_relative))
            print ("{new_utt} {utt_id} {s} {e}".format(
                new_utt=new_utt, utt_id=utt_id, s=start_relative,
                e=start_relative + this_dur))
            start += this_dur - args.overlap_duration
            dur -= this_dur - args.overlap_duration
            this_dur = positive_dur[random.randrange(num_pos_utts)]

        new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
            utt_id=utt_id, s=int(round(100 * (start - start_time))),
            e=int(round(100 * (end_time - start_time))))
        print ("{new_utt} {utt_id} {s} {e}".format(
            new_utt=new_utt, utt_id=utt_id, s=start - start_time,
            e=end_time - start_time))


def main():
    args = get_args()
    try:
        run(args)
    except Exception:
        logging.error("Failed creating subsegments", exc_info=True)
        raise SystemExit(1)
    finally:
        args.segments_file.close()


if __name__ == '__main__':
    main()
