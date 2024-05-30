#!/usr/bin/env python

from __future__ import print_function
import sys
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser(description="""
 Usage: extend_segment_times.py [options] <input-segments >output-segments
 This program pads the times in a 'segments' file (e.g. data/train/segments)
 with specified left and right context (for cases where there was no
 silence padding in the original segments file)""")

parser.add_argument("--start-padding", type = float, default = 0.1,
                    help="Amount of padding, in seconds, for the start time of "
                    "each segment (start times <0 will be set to zero).")
parser.add_argument("--end-padding", type = float, default = 0.1,
                    help="Amount of padding, in seconds, for the end time of "
                    "each segment.")
parser.add_argument("--last-segment-end-padding", type = float, default = 0.1,
                    help="Amount of padding, in seconds, for the end time of "
                    "the last segment of each file (maximum allowed).")
parser.add_argument("--fix-overlapping-segments", type = str,
                    default = 'true', choices=['true', 'false'],
                    help="If true, prevent segments from overlapping as a result "
                    "of the padding (or that were already overlapping)")
args = parser.parse_args()


# the input file will be a sequence of lines which are each of the form:
# <utterance-id> <recording-id> <start-time> <end-time>
# e.g.
# utt-1 recording-1 0.62 5.40
# The output will be in the same format and in the same
# order, except wiht modified times.

# This variable maps from a recording-id to a listof the utterance
# indexes (as integer indexes into 'entries']
# that are part of that recording.
recording_to_utt_indexes = defaultdict(list)

# This is an array of the entries in the segments file, in the fomrat:
# (utterance-id as astring, recording-id as string,
#  start-time as float, end-time as float)
entries = []


while True:
    line = sys.stdin.readline()
    if line == '':
        break
    try:
        [ utt_id, recording_id, start_time, end_time ] = line.split()
        start_time = float(start_time)
        end_time = float(end_time)
    except:
        sys.exit("extend_segment_times.py: could not interpret line: " + line)
    if not end_time > start_time:
        print("extend_segment_times.py: bad segment (ignoring): " + line,
              file = sys.stderr)
    recording_to_utt_indexes[recording_id].append(len(entries))
    entries.append([utt_id, recording_id, start_time, end_time])

num_times_fixed = 0

for recording, utt_indexes in recording_to_utt_indexes.items():
    # this_entries is a list of lists, sorted on mid-time.
    # Notice: because lists are objects, when we change 'this_entries'
    # we change the underlying entries.
    this_entries = sorted([ entries[x] for x in utt_indexes ],
                          key = lambda x : 0.5 * (x[2] + x[3]))
    min_time = 0
    max_time = max([ x[3] for x in this_entries ]) + args.last_segment_end_padding
    start_padding = args.start_padding
    end_padding = args.end_padding
    for n in range(len(this_entries)):
        this_entries[n][2] = max(min_time, this_entries[n][2] - start_padding)
        this_entries[n][3] = min(max_time, this_entries[n][3] + end_padding)

    for n in range(len(this_entries) - 1):
        this_end_time = this_entries[n][3]
        next_start_time = this_entries[n+1][2]
        if this_end_time > next_start_time and args.fix_overlapping_segments == 'true':
            midpoint = 0.5 * (this_end_time + next_start_time)
            this_entries[n][3] = midpoint
            this_entries[n+1][2] = midpoint
            num_times_fixed += 1


# this prints a number with a certain number of digits after
# the point, while removing trailing zeros.
def FloatToString(f):
    num_digits = 6 # we want to print 6 digits after the zero
    g = f
    while abs(g) > 1.0:
        g *= 0.1
        num_digits += 1
    format_str = '%.{0}g'.format(num_digits)
    return format_str % f

for entry in entries:
    [ utt_id, recording_id, start_time, end_time ] = entry
    if not start_time < end_time:
        print("extend_segment_times.py: bad segment after processing (ignoring): " +
              ' '.join(entry), file = sys.stderr)
        continue
    print(utt_id, recording_id, FloatToString(start_time), FloatToString(end_time))


print("extend_segment_times.py: extended {0} segments; fixed {1} "
      "overlapping segments".format(len(entries), num_times_fixed),
      file = sys.stderr)

## test:
#  (echo utt1 reco1 0.2 6.2; echo utt2 reco1 6.3 9.8 )| extend_segment_times.py
# and also try the above with the options --last-segment-end-padding=0.0 --fix-overlapping-segments=false

