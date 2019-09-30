#! /usr/bin/env python

# Copyright 2017  Vimal Manohar
#           2017  Matthew Maciejewski
# Apache 2.0.

from __future__ import print_function
import argparse
import logging
import sys
import textwrap

def get_args():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
        Creates a subsegments file from an input segments file.

        The output format is

        <subsegment-id> <utterance-id> <start-time> <end-time>,

        where the timings are relative to the start-time of the
         <utterance-id> in the input segments file.
        Reminder: the format of the input segments file is:

         <utterance-id> <recording-id> <start-time> <end-time>

        where the recording-id corresponds to a wav file (or a channel of
        a wav file) from wav.scp.  Note: you can use
        utils/data/get_segments_for_data.sh to generate a 'default'
        segments file for your data if one doesn't already exist.

        e.g.: get_uniform_subsegments.py data/dev/segments > \\
                data/dev_uniform_segments/sub_segments

        utils/data/subsegment_data_dir.sh data/dev \\
            data/dev_uniform_segments/sub_segments data/dev_uniform_segments

        The output is written to stdout. The resulting file can be
        passed to utils/data/subsegment_data_dir.sh to sub-segment
        the data directory."""),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-segment-duration", type=float,
                        default=30, help="""Maximum duration of the
                        subsegments (in seconds)""")
    parser.add_argument("--overlap-duration", type=float,
                        default=5, help="""Overlap between
                        adjacent segments (in seconds)""")
    parser.add_argument("--max-remaining-duration", type=float,
                        default=10, help="""Segment is not split
                        if the left-over duration is more than this
                        many seconds""")
    parser.add_argument("--constant-duration", type=bool,
                        default=False, help="""Final segment is given
                        a start time max-segment-duration before the
                        end to force a constant segment duration. This
                        overrides the max-remaining-duration parameter""")
    parser.add_argument("segments_file", type=argparse.FileType('r'),
                        help="""Input kaldi segments file""")

    args = parser.parse_args()
    return args


def run(args):
    if (args.constant_duration):
        dur_threshold = args.max_segment_duration
    else:
        dur_threshold = args.max_segment_duration + args.max_remaining_duration

    for line in args.segments_file:
        parts = line.strip().split()
        utt_id = parts[0]
        start_time = float(parts[2])
        end_time = float(parts[3])

        dur = end_time - start_time

        start = start_time
        while (dur > dur_threshold):
            end = start + args.max_segment_duration
            start_relative = start - start_time
            end_relative = end - start_time
            new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
                utt_id=utt_id, s=int(100 * start_relative),
                e=int(100 * end_relative))
            print ("{new_utt} {utt_id} {s:.3f} {e:.3f}".format(
                new_utt=new_utt, utt_id=utt_id, s=start_relative,
                e=start_relative + args.max_segment_duration))
            start += args.max_segment_duration - args.overlap_duration
            dur -= args.max_segment_duration - args.overlap_duration

        if (args.constant_duration):
            if (dur < 0):
              continue
            if (dur < args.max_remaining_duration):
              start = max(end_time - args.max_segment_duration, start_time)
            end = min(start + args.max_segment_duration, end_time)
        else:
            end = end_time
        new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
            utt_id=utt_id, s=int(round(100 * (start - start_time))),
            e=int(round(100 * (end - start_time))))
        print ("{new_utt} {utt_id} {s:.3f} {e:.3f}".format(
            new_utt=new_utt, utt_id=utt_id, s=start - start_time,
            e=end - start_time))


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
