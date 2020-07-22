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
with open(os.path.join(out_dir, "utt2spk"), 'w') as utt2spk_fi , open(os.path.join(out_dir, "segments"), 'w') as segments_fi: 
    for subdir, dirs, files in os.walk(wav_dir):
      for file in files:
        utt = str(file).replace(".sph", "")
        if file.endswith(".sph") and utt in utts:
          wav = "{0}{1} sox {2}/{1}.sph -c 1 -r 16000 -t wav - |\n".format(wav, utt, subdir)
    wav_fi = open(os.path.join(out_dir, "wav.scp"), 'w')
    wav_fi.write(wav)

    for utt in utts:
      music_filename = utt + "_music.key.refined"
      speech_filename = utt + "_speech.key.refined"
      music_fi = open(os.path.join(out_dir, music_filename), 'r').readlines()
      speech_fi = open(os.path.join(out_dir, speech_filename), 'r').readlines()
      for count, line in enumerate(music_fi, 1):
        left, right = line.rstrip().split(" ")
        segments_fi.write("{0}-music-{1} {0} {2} {3}\n".format(utt, count, left, right))
        utt2spk_fi.write("{0}-music-{1} {0}-music-{1}\n".format(utt,count))
      for count, line in enumerate(speech_fi, 1):
        left, right = line.rstrip().split(" ")
        segments_fi.write("{0}-speech-{1} {0} {2} {3}\n".format(utt, count, left, right))
        utt2spk_fi.write("{0}-speech-{1} {0}-speech-{1}\n".format(utt,count))

