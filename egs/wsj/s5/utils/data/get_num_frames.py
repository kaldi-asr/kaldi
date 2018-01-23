#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse
parser = argparse.ArgumentParser(description="""
         Calculate the total frame number from utt2dur file.""")

parser.add_argument("--frame-shift", type = float, default = 0.01)
parser.add_argument("--utt2dur-dir", type = str, default = "")
args = parser.parse_args()

utt2dur_file = open(args.utt2dur_dir)
wav_time = 0.0

while 1:
  line = utt2dur_file.readline()
  if not line:
    break
  wav_time += float(line.split()[1])

num_frames = wav_time / args.frame_shift
print(str(num_frames).split('.')[0])

