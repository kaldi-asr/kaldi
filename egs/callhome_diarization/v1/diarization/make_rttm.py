#!/usr/bin/env python3

# Copyright  2016  David Snyder
#            2017  Matthew Maciejewski
# Apache 2.0.

"""This script converts a segments and labels file to a NIST RTTM
file. It creates flat segmentation (i.e. no overlapping regions)
from overlapping segments, e.g. the output of a sliding-window
diarization system. The speaker boundary between two overlapping
segments by different speakers is placed at the midpoint between
the end of the first segment and the start of the second segment.

The segments file format is:
<segment-id> <recording-id> <start-time> <end-time>
The labels file format is:
<segment-id> <speaker-id>

The output RTTM format is:
<type> <file> <chnl> <tbeg> \
        <tdur> <ortho> <stype> <name> <conf> <slat>
where:
<type> = "SPEAKER"
<file> = <recording-id>
<chnl> = "0"
<tbeg> = start time of segment
<tdur> = duration of segment
<ortho> = "<NA>"
<stype> = "<NA>"
<name> = <speaker-id>
<conf> = "<NA>"
<slat> = "<NA>"
"""

import argparse
import sys
import codecs


def get_args():
  parser = argparse.ArgumentParser(
    description="""This script converts a segments and labels file
    to a NIST RTTM file. It handles overlapping segments (e.g. the
    output of a sliding-window diarization system).""")

  parser.add_argument("segments", type=str,
                      help="Input segments file")
  parser.add_argument("labels", type=str,
                      help="Input labels file")
  parser.add_argument("rttm_file", type=str,
                      help="Output RTTM file")
  parser.add_argument("--rttm-channel", type=int, default=0,
                      help="The value passed into the RTTM channel field. \
                      Only affects the format of the RTTM file.")

  args = parser.parse_args()
  return args

def main():
  args = get_args()

  # File containing speaker labels per segment
  seg2label = {}
  with codecs.open(args.labels, 'r', 'utf-8') as labels_file:
    for line in labels_file:
      seg, label = line.strip().split()
      seg2label[seg] = label

  # Segments file
  reco2segs = {}
  with codecs.open(args.segments, 'r', 'utf-8') as segments_file:
    for line in segments_file:
      seg, reco, start, end = line.strip().split()
      try:
        if reco in reco2segs:
          reco2segs[reco] = reco2segs[reco] + " " + start + "," + end + "," + seg2label[seg]
        else:
          reco2segs[reco] = reco + " " + start + "," + end + "," + seg2label[seg]
      except KeyError:
        raise RuntimeError("Missing label for segment {0}".format(seg))

  # Cut up overlapping segments so they are contiguous
  contiguous_segs = []
  for reco in sorted(reco2segs):
    segs = reco2segs[reco].strip().split()
    new_segs = ""
    for i in range(1, len(segs)-1):
      start, end, label = segs[i].split(',')
      next_start, next_end, next_label = segs[i+1].split(',')
      if float(end) > float(next_start):
        done = False
        avg = str((float(next_start) + float(end)) / 2.0)
        segs[i+1] = ','.join([avg, next_end, next_label])
        new_segs += " " + start + "," + avg + "," + label
      else:
        new_segs += " " + start + "," + end + "," + label
    start, end, label = segs[-1].split(',')
    new_segs += " " + start + "," + end + "," + label
    contiguous_segs.append(reco + new_segs)

  # Merge contiguous segments of the same label
  merged_segs = []
  for reco_line in contiguous_segs:
    segs = reco_line.strip().split()
    reco = segs[0]
    new_segs = ""
    for i in range(1, len(segs)-1):
      start, end, label = segs[i].split(',')
      next_start, next_end, next_label = segs[i+1].split(',')
      if float(end) == float(next_start) and label == next_label:
        segs[i+1] = ','.join([start, next_end, next_label])
      else:
        new_segs += " " + start + "," + end + "," + label
    start, end, label = segs[-1].split(',')
    new_segs += " " + start + "," + end + "," + label
    merged_segs.append(reco + new_segs)

  with codecs.open(args.rttm_file, 'w', 'utf-8') as rttm_writer:
    for reco_line in merged_segs:
      segs = reco_line.strip().split()
      reco = segs[0]
      for i in range(1, len(segs)):
        start, end, label = segs[i].strip().split(',')
        print("SPEAKER {0} {1} {2:7.3f} {3:7.3f} <NA> <NA> {4} <NA> <NA>".format(
          reco, args.rttm_channel, float(start), float(end)-float(start), label), file=rttm_writer)

if __name__ == '__main__':
  main()
