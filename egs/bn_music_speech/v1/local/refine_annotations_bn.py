#!/usr/bin/env python
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script refines the annotation files produced by
# make_annotations_bn.py. In order to create unambiguous annotations,
# we remove any part of a segment that overlaps with another. Also,
# this script merges together contiguous segments that have the
# same annotation, and ensures that only segments longer than a
# designated length are created.
#
# This file is meant to be invoked from make_bn.sh.
from __future__ import division
import sys, os

def seg_to_string(seg):
  start = seg[0]
  end = seg[1]
  if start < end:
    return str(start) + " " + str(end) + "\n"
  else:
    return ""

def process_segs(raw_segs):
  segs = []
  for seg in raw_segs:
    lower, upper = [float(i) for i in seg.rstrip().split(" ")]
    segs.append((lower, upper))
  return segs

def resegment(music, speech, other, frame_length, min_seg):
  frame2classes = []
  max_duration = 0
  all_segs = music + speech + other
  for (start, end) in all_segs:
    if end > max_duration:
      max_duration = end
  num_frames = int(max_duration) * frame_length
  for i in range(0, num_frames):
    frame2classes.append([])

  annotate_frames(frame2classes, music, "music", frame_length, num_frames)
  annotate_frames(frame2classes, speech, "speech", frame_length, num_frames)
  annotate_frames(frame2classes, other,  "other", frame_length, num_frames)

  curr_class = None
  for i in range(0, len(frame2classes)):
    if len(frame2classes[i]) != 1 or frame2classes[i][0] == "other":
      curr_class = "other"
    elif frame2classes[i][0] == "music":
      curr_class = "music"
    elif frame2classes[i][0] == "speech":
      curr_class = "speech"
    else:
      curr_class = "other"
    frame2classes[i] = curr_class

  new_music = []
  new_speech = []
  curr_class = frame2classes[0]
  start_frame = 0
  for i in range(1, len(frame2classes)):
    if curr_class != frame2classes[i]:
      start = float(start_frame)/frame_length
      end = float(i)/frame_length
      if end - start > min_seg:
        if curr_class == "music":
          new_music.append((start, end))
        elif curr_class == "speech":
          new_speech.append((start, end))
      start_frame = i
      curr_class = frame2classes[i]

  return new_music, new_speech


def annotate_frames(frame2classes, segs, annotation, frame_length, max_duration):
  for (start, end) in segs:
    frame_start = min(int(start * frame_length), max_duration)
    frame_end = min(int(end * frame_length), max_duration)
    for i in range(frame_start, frame_end):
      frame2classes[i].append(annotation)

def main():
  out_dir = sys.argv[1]
  frames_per_sec = int(sys.argv[2])
  min_seg_length = float(sys.argv[3])

  with open(os.path.join(out_dir, "utt_list"), 'r') as utts:
    for line in utts:
      speech_filename = os.path.join(out_dir, line.rstrip() + "_speech.key")
      music_filename = os.path.join(out_dir, line.rstrip() + "_music.key")
      other_filename = os.path.join(out_dir, line.rstrip() + "_other.key")
      speech_fi = open(speech_filename, 'r')
      raw_speech_segs = speech_fi.readlines()
      speech_fi.close()
      music_fi = open(music_filename, 'r')
      raw_music_segs = music_fi.readlines()
      music_fi.close()
      other_fi = open(other_filename, 'r')
      raw_other_segs = other_fi.readlines()
      other_fi.close()
      speech_segs = process_segs(raw_speech_segs)
      music_segs = process_segs(raw_music_segs)
      other_segs = process_segs(raw_other_segs)
      music_segs, speech_segs = resegment(music_segs, speech_segs, other_segs, frames_per_sec, min_seg_length)

      with open(speech_filename + ".refined", 'w') as speech_fi, \
          open(music_filename + ".refined", 'w') as music_fi:
        for seg in music_segs:
          music_fi.write(seg_to_string(seg))
        for seg in speech_segs:
          speech_fi.write(seg_to_string(seg))


if __name__=="__main__":
  main()
