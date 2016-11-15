#!/bin/bash

# Copyright  2016  David Snyder
# Apache 2.0.
# TODO, script needs some work, error handling, etc

import sys

segments_fi = open(sys.argv[1], 'r').readlines()
label_fi = open(sys.argv[2], 'r').readlines()

# File containing speaker labels per utt
seg2label = {}
for l in label_fi:
  seg, label = l.rstrip().split()
  seg2label[seg] = label

# Segments file
utt2seg = {}
for l in segments_fi:
  seg, utt, s, e = l.rstrip().split()
  if utt in utt2seg:
    utt2seg[utt] = utt2seg[utt] + " " + s + "," + e + "," + seg2label[seg]
  else:
    utt2seg[utt] = utt + " " + s + "," + e + "," + seg2label[seg]


# TODO Cut up the segments so that they are contiguous
diarization1 = []
for utt in utt2seg:
  l = utt2seg[utt]
  t = l.rstrip().split()
  utt = t[0]
  rhs = ""
  for i in range(1,len(t)-1):
    s, e, label = t[i].split(',')
    s_next, e_next, label_next = t[i+1].split(',')
    if float(e) > float(s_next):
      avg = str((float(s_next) + float(e)) / 2.0)
      t[i+1] = ','.join([avg, e_next, label_next])
      rhs += " " + s + "," + avg + "," + label
    else:
      rhs += " " + s + "," + e + "," + label
  s, e, label = t[-1].split(',')
  rhs += " " + s + "," + e + "," + label
  diarization1.append(utt + rhs)

# TODO Merge the contiguous segments that belong to the same speaker
diarization2 = []
for l in diarization1:
  t = l.rstrip().split()
  utt = t[0]
  rhs = ""
  for i in range(1,len(t)-1):
    s, e, label = t[i].split(',')
    s_next, e_next, label_next = t[i+1].split(',')
    if float(e) == float(s_next) and label == label_next:
      t[i+1] = ','.join([s, e_next, label_next])
    else:
      rhs += " " + s + "," + e + "," + label
  s, e, label = t[-1].split(',')
  rhs += " " + s + "," + e + "," + label
  diarization2.append(utt + rhs)


for l in diarization2:
  t = l.rstrip().split()
  utt = t[0]
  for i in range(1, len(t)):
    s, e, label = t[i].rstrip().split(',')
    print "SPEAKER", utt, 0, s, float(e) - float(s), "<NA> <NA>", label, "<NA> <NA>"


