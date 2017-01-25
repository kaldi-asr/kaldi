#!/usr/bin/env python
# Copyright 2014  Johns Hopkins University (Authors: Daniel Povey, Vijayaditya Peddinti).
#           2016  Vimal Manohar
# Apache 2.0.

# Script to combine ctms with overlapping segments

import sys, math, numpy as np, argparse
break_threshold = 0.01

def ReadSegments(segments_file):
  segments = {}
  for line in open(segments_file).readlines():
    parts = line.strip().split()
    segments[parts[0]] = (parts[1], float(parts[2]), float(parts[3]))
  return segments

#def get_breaks(ctm, prev_end):
#  breaks = []
#  for i in xrange(0, len(ctm)):
#    if ctm[i][2] - prev_end > break_threshold:
#      breaks.append([i, ctm[i][2]])
#    prev_end = ctm[i][2] + ctm[i][3]
#  return np.array(breaks)

# Resolve overlaps within segments of the same recording
def ResolveOverlaps(ctms, segments):
  total_ctm = []
  if len(ctms) == 0:
    raise Exception('Something wrong with the input ctms')

  next_utt = ctms[0][0][0]
  for ctm_index in range(len(ctms) - 1):
    # Assumption here is that the segments are written in consecutive order?
    cur_ctm = ctms[ctm_index]
    next_ctm = ctms[ctm_index + 1]

    cur_utt = next_utt
    next_utt = next_ctm[0][0]
    if (next_utt not in segments):
      raise Exception('Could not find utterance %s in segments' % next_utt)

    if len(cur_ctm) > 0:
      assert(cur_utt == cur_ctm[0][0])

    assert(next_utt > cur_utt)
    if (cur_utt not in segments):
      raise Exception('Could not find utterance %s in segments' % cur_utt)

    # length of this segment
    window_length = segments[cur_utt][2] - segments[cur_utt][1]

    # overlap of this segment with the next segment
    # Note: It is possible for this to be negative when there is actually
    # no overlap between consecutive segments.
    overlap = segments[cur_utt][2] - segments[next_utt][1]

    # find the breaks after overlap starts
    index = len(cur_ctm)

    for i in xrange(len(cur_ctm)):
      if (cur_ctm[i][2] + cur_ctm[i][3]/2.0 > (window_length - overlap/2.0)):
        # if midpoint of a hypothesis word is beyond the midpoint of the
        # overlap region
        index = i
        break

    # Ignore the hypotheses beyond this midpoint. They will be considered as
    # part of the next segment.
    total_ctm += cur_ctm[:index]

    # Ignore the hypotheses of the next utterance that overlaps with the
    # current utterance
    index = -1
    for i in xrange(len(next_ctm)):
      if (next_ctm[i][2] + next_ctm[i][3]/2.0 > (overlap/2.0)):
        index = i
        break

    if index >= 0:
        ctms[ctm_index + 1] = next_ctm[index:]
    else:
        ctms[ctm_index + 1] = []

  # merge the last ctm entirely
  total_ctm += ctms[-1]

  return total_ctm

def ReadCtm(ctm_file_lines, segments):
  ctms = {}
  for key in [ x[0] for x in segments.values() ]:
    ctms[key] = []

  ctm = []
  prev_utt = ctm_file_lines[0].split()[0]
  for line in ctm_file_lines:
    parts = line.split()
    if (prev_utt == parts[0]):
      ctm.append([parts[0], parts[1], float(parts[2]),
        float(parts[3])] + parts[4:])
    else:
      # New utterance. Append the previous utterance's CTM
      # into the list for the utterance's recording
      ctms[segments[ctm[0][0]][0]].append(ctm)

      assert(parts[0] > prev_utt)

      prev_utt = parts[0]
      ctm = []
      ctm.append([parts[0], parts[1], float(parts[2]),
        float(parts[3])] + parts[4:])

  # append the last ctm
  ctms[segments[ctm[0][0]][0]].append(ctm)
  return ctms

def WriteCtm(ctm_lines, out_file):
  for line in ctm_lines:
    out_file.write("{0} {1} {2} {3} {4}\n".format(line[0], line[1], line[2], line[3], " ".join(line[4:])))

if __name__ == "__main__":
  usage = """ Python script to resolve overlaps in ctms """
  parser = argparse.ArgumentParser(usage)
  parser.add_argument('segments', type=str, help = 'use segments to resolve overlaps')
  parser.add_argument('ctm_in', type=str, help='input_ctm_file')
  parser.add_argument('ctm_out', type=str, help='output_ctm_file')
  params = parser.parse_args()

  if params.ctm_in == "-":
    params.ctm_in = sys.stdin
  else:
    params.ctm_in = open(params.ctm_in)
  if params.ctm_out == "-":
    params.ctm_out = sys.stdout
  else:
    params.ctm_out = open(params.ctm_out, 'w')

  segments = ReadSegments(params.segments)

  # Read CTMs into a dictionary indexed by the recording
  ctms = ReadCtm(params.ctm_in.readlines(), segments)

  for key in sorted(ctms.keys()):
    # Process CTMs in the sorted order of recordings
    ctm_reco = ctms[key]
    ctm_reco = ResolveOverlaps(ctm_reco, segments)
    WriteCtm(ctm_reco, params.ctm_out)
  params.ctm_out.close()
