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

with open(os.path.join(out_dir, "utt2spk"), 'w') as utt2spk_fi, \
    open(os.path.join(out_dir, "segments"), 'w') as segments_fi, \
    open(os.path.join(out_dir, "wav.scp"), 'w') as wav_fi, \
    open(os.path.join(out_dir, "utt_list"), 'r') as utt_fi:
  utts = set(utt.rstrip() for utt in utt_fi.readlines())
  for subdir, dirs, files in os.walk(wav_dir):
    for f in files:
      utt = str(f).replace(".sph", "")
      if f.endswith(".sph") and utt in utts:
        wav_fi.write("{0} sox {1}/{0}.sph -c 1 -r 16000 -t wav - |\n".format(utt, subdir))

    for utt in utts:
      music_filename = utt + "_music.key.refined"
      speech_filename = utt + "_speech.key.refined"
      with open(os.path.join(out_dir, music_filename), 'r') as music_fi, \
          open(os.path.join(out_dir, speech_filename), 'r') as speech_fi:
        for count, line in enumerate(music_fi, 1):
          left, right = line.rstrip().split(" ")
          segments_fi.write("{0}-music-{1} {0} {2} {3}\n".format(utt, count, left, right))
          utt2spk_fi.write("{0}-music-{1} {0}-music-{1}\n".format(utt,count))
        for count, line in enumerate(speech_fi, 1):
          left, right = line.rstrip().split(" ")
          segments_fi.write("{0}-speech-{1} {0} {2} {3}\n".format(utt, count, left, right))
          utt2spk_fi.write("{0}-speech-{1} {0}-speech-{1}\n".format(utt,count))

