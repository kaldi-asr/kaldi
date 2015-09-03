#!/usr/bin/env python
# Copyright 2014  Johns Hopkins University (Authors: Daniel Povey, Vijayaditya Peddinti).  Apache 2.0.

# Script to combine ctms for uniformly segmented, with overlaps

import sys, math, numpy as np, argparse
break_threshold = 0.01
def get_breaks(ctm, prev_end):
  breaks = []
  for i in xrange(0, len(ctm)):
    if ctm[i][2] - prev_end > break_threshold:
      breaks.append([i, ctm[i][2]])
    prev_end = ctm[i][2] + ctm[i][3]
  return np.array(breaks)

def resolve_overlaps(ctms, window_length, overlap):
  total_ctm = []
  if len(ctms) == 0:
    raise Exception('Something wrong with the input ctms')
  for ctm_index in range(len(ctms) - 1):
    cur_ctm = ctms[ctm_index]
    next_ctm = ctms[ctm_index + 1]
    # find the breaks after overlap starts
    index = len(cur_ctm)
    for i in xrange(len(cur_ctm)):
      if cur_ctm[i][2] + cur_ctm[i][3]/2.0 > (window_length - overlap/2.0):
        index = i
        break
    total_ctm += cur_ctm[:index]
    
    index = 0
    for i in xrange(len(next_ctm)):
      if next_ctm[i][2] + next_ctm[i][3]/2.0 > (overlap/2.0):
        index = i
        break
    ctms[ctm_index + 1] = next_ctm[index:]
  # merge the last ctm entirely
  total_ctm +=ctms[-1]
  return total_ctm
def read_ctm(ctm_file_lines, utt2spk):
  ctms = {}
  for key in utt2spk.values():
    ctms[key] = []

  ctm = []
  prev_utt = ctm_file_lines[0].split()[0]
  for line in ctm_file_lines:
    parts = line.split()
    if prev_utt == parts[0]:
      ctm.append([parts[0], parts[1], float(parts[2]),
        float(parts[3]), parts[4], parts[5]])
    else:
      ctms[utt2spk[ctm[0][0]]].append(ctm)
      prev_utt = parts[0]
      ctm = []
      ctm.append([parts[0], parts[1], float(parts[2]),
        float(parts[3]), parts[4], parts[5]])
  # append the last ctm
  ctms[utt2spk[ctm[0][0]]].append(ctm) 
  return ctms

def write_ctm(ctm_lines):
  ctm_file_lines = []
  for line in ctm_lines:
    ctm_file_lines.append("{0} {1} {2} {3} {4} {5}".format(line[0],line[1],line[2],line[3],line[4],line[5]))
  return ctm_file_lines

if __name__ == "__main__":
  usage = """ Python script to resolve overlaps in uniformly segmented ctms """ 
  main_parser = argparse.ArgumentParser(usage)
  parser = argparse.ArgumentParser()
  parser.add_argument('--window-length', type = float, default = 30.0, help = 'length of the window used to cut the segment')
  parser.add_argument('--overlap', type = float, default = 5.0, help = 'overlap of neighboring windows')
  parser.add_argument('utt2spk', type=str, help='spk2utt_file')
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

  utt2spk = {}
  for line in open(params.utt2spk).readlines():
    parts = line.split()
    utt2spk[parts[0]] = parts[1]

  ctms = read_ctm(params.ctm_in.readlines(), utt2spk)
  speakers = ctms.keys()
  speakers.sort()
  for key in speakers:
    ctm = ctms[key]
    ctm = resolve_overlaps(ctm, params.window_length, params.overlap)
    params.ctm_out.write("\n".join(write_ctm(ctm))+"\n")
  params.ctm_out.close()
