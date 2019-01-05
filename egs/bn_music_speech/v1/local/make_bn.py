#!/usr/bin/env python
# Copyright 2015   David Snyder
# Apache 2.0.
#
# Using the annotations created by refine_annotations_bn.py, this script
# creates the segments, utt2spk, and wav.scp files.
#
# This file is meant to be invoked by make_bn.sh.

import os, sys
wav_dir = sys.argv[1]
out_dir = sys.argv[2]

utts = open(os.path.join(out_dir, "utt_list"), 'r').readlines()
utts = set(x.rstrip() for x in utts)
wav = ""
segments = ""
utt2spk = ""
for subdir, dirs, files in os.walk(wav_dir):
  for file in files:
    utt = str(file).replace(".sph", "")
    if file.endswith(".sph") and utt in utts:
      wav = "{0}{1} sox {2}/{1}.sph -c 1 -r 16000 -t -wav - |\n".format(wav, utt, subdir)
wav_fi = open(os.path.join(out_dir, "wav.scp"), 'w')
wav_fi.write(wav)

for utt in utts:
  music_filename = utt + "_music.key.refined"
  speech_filename = utt + "_speech.key.refined"
  music_fi = open(os.path.join(out_dir, music_filename), 'r').readlines()
  speech_fi = open(os.path.join(out_dir, speech_filename), 'r').readlines()
  count = 1
  for line in music_fi:
    left, right = line.rstrip().split(" ")
    segments = "{0}{1}-music-{2} {1} {3} {4}\n".format(segments, utt, count, left, right)
    utt2spk = "{0}{1}-music-{2} {1}-music-{2}".format(utt2spk, utt,count)
    count += 1
  count = 1
  for line in speech_fi:
    left, right = line.rstrip().split(" ")
    segments = "{0}{1}-speech-{2} {1} {3} {4}\n".format(segments, utt, count, left, right)
    utt2spk = "{0}{1}-speech-{2} {1}-music-{2}".format(utt2spk, utt, count)
    count += 1
utt2spk_fi = open(os.path.join(out_dir, "utt2spk"), 'w')
utt2spk_fi.write(utt2spk)
segments_fi = open(os.path.join(out_dir, "segments"), 'w')
segments_fi.write(segments)

