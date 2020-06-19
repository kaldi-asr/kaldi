#!/usr/bin/env python

# Copyright  2020  Desh Raj
# Apache 2.0.

"""This script converts a segments and labels file to a NIST RTTM
file. It can handle overlapping segmentation.

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
import itertools

from intervaltree import Interval, IntervalTree

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

def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def merge_segments(segs):
    """Merge overlapping segments by same speaker in a recording."""
    # Merge separately for each speaker.
    new_segs = []
    for speaker_id, speaker_segs in groupby(segs, lambda x: x[2]):
        speaker_segs = list(speaker_segs)
        speaker_it = IntervalTree.from_tuples([(seg[0], seg[1]) for seg in speaker_segs])
        n_segs_pre = len(speaker_it)
        speaker_it.merge_overlaps(strict=False)
        n_segs_post = len(speaker_it)
        if n_segs_post < n_segs_pre:
            speaker_segs = []
            for intrvl in speaker_it:
                speaker_segs.append((intrvl.begin, intrvl.end, speaker_id))
        new_segs.extend(speaker_segs)
    return new_segs

def main():
  args = get_args()

  # File containing speaker labels per segment
  seg2label = {}
  with open(args.labels, 'r') as labels_file:
    for line in labels_file:
      seg, label = line.strip().split()
      seg2label[seg] = label

  # Segments file
  reco2segs = {}
  with open(args.segments, 'r') as segments_file:
    for line in segments_file:
      seg, reco, start, end = line.strip().split()
      try:
        if reco in reco2segs:
          reco2segs[reco].append((float(start), float(end), seg2label[seg]))
        else:
          reco2segs[reco] = [(float(start), float(end), seg2label[seg])]
      except KeyError:
        raise RuntimeError("Missing label for segment {0}".format(seg))

  # Merge overlapping subsegments of the same speaker in a recording
  reco2merged_segs = {}
  for reco in sorted(reco2segs):
    merged_segs = merge_segments(reco2segs[reco])
    reco2merged_segs[reco] = merged_segs

  with open(args.rttm_file, 'w') as rttm_writer:
    for reco in reco2merged_segs:
      segs = reco2merged_segs[reco]
      for seg in sorted(segs, key=lambda x: x[0]):
        start, end, label = seg
        rttm_writer.write("SPEAKER {0} {1} {2:7.3f} {3:7.3f} <NA> <NA> {4} <NA> <NA>\n".format(
          reco, args.rttm_channel, start, end-start, label))

if __name__ == '__main__':
  main()
