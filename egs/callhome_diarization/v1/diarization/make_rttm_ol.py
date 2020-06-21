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

class Segment:
    """Stores all information about a segment"""
    def __init__(self, reco_id, start_time, end_time, labels):
        self.reco_id = reco_id
        self.start_time = start_time
        self.end_time = end_time
        self.dur = end_time - start_time
        self.labels = labels

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
  with open(args.labels, 'r') as labels_file:
    for line in labels_file:
      seg, label = line.strip().split()
      if seg in seg2label:
        seg2label[seg].append(label)
      else:
        seg2label[seg] = [label]

  # Segments file
  reco2segs = {}
  with open(args.segments, 'r') as segments_file:
    for line in segments_file:
      seg, reco, start, end = line.strip().split()
      try:
        if reco in reco2segs:
          reco2segs[reco].append(Segment(reco, float(start), float(end), seg2label[seg]))
        else:
          reco2segs[reco] = [Segment(reco, float(start), float(end), seg2label[seg])]
      except KeyError:
        raise RuntimeError("Missing label for segment {0}".format(seg))

  # At this point the subsegments are overlapping, since we got them from a
  # sliding window diarization method. We make them contiguous here
  reco2contiguous = {}
  for reco in sorted(reco2segs):
    segs = sorted(reco2segs[reco], key=lambda x: x.start_time)
    new_segs = []
    for i, seg in enumerate(segs):
      # If it is last segment in recording or last contiguous segment, add it to new_segs
      if (i == len(segs)-1 or seg.end_time <= segs[i+1].start_time):
        new_segs.append(Segment(reco, seg.start_time, seg.end_time, seg.labels))
      # Otherwise split overlapping interval between current and next segment
      else:
        avg = (segs[i+1].start_time + seg.end_time) / 2
        new_segs.append(Segment(reco, seg.start_time, avg, seg.labels))
        segs[i+1].start_time = avg
    reco2contiguous[reco] = new_segs

  # Merge contiguous segments of the same label
  reco2merged = {}
  for reco in reco2contiguous:
    segs = reco2contiguous[reco]
    new_segs = []
    running_labels = {} # {label: (start_time, end_time)}
    for i, seg in enumerate(segs):
      # If running labels are not present in current segment, add those segments
      # to new_segs list and delete those entries
      for label in list(running_labels):
        if label not in seg.labels:
          new_segs.append(Segment(reco, running_labels[label][0], running_labels[label][1], label))
          del running_labels[label]
      # Now add/update labels in running_labels based on current segment
      for label in seg.labels:
        if label in running_labels:
          # If already present, just update end time
          start_time = running_labels[label][0]
          running_labels[label] = (start_time, seg.end_time)
        else:
          # Otherwise make a new entry
          running_labels[label] = (seg.start_time, seg.end_time)
      # If it is the last segment in utterance or last contiguous segment, add it to new_segs
      # and delete from running_labels
      if (i == len(segs)-1 or seg.end_time < segs[i+1].start_time):
        # Case when it is last segment or if next segment is after some gap
        for label in list(running_labels):
          new_segs.append(Segment(reco,  running_labels[label][0], running_labels[label][1], label))
          del running_labels[label]
    reco2merged[reco] = new_segs

  with open(args.rttm_file, 'w') as rttm_writer:
    for reco in reco2merged:
      segs = reco2merged[reco]
      for seg in segs:
        for label in seg.labels:
          rttm_writer.write("SPEAKER {0} {1} {2:7.3f} {3:7.3f} <NA> <NA> {4} <NA> <NA>\n".format(
            reco, args.rttm_channel, seg.start_time, seg.dur, label))

if __name__ == '__main__':
  main()
