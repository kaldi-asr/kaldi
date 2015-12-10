#!/usr/bin/env python
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This file is meant to be invoked by make_musan.sh.

import os, sys


def process_music_annotations(path):
  utt2spk = {}
  utt2vocals = {}
  lines = open(path, 'r').readlines()
  for line in lines:
    utt, genres, vocals, musician = line.rstrip().split()[:4]
    # For this application, the musican ID isn't important
    utt2spk[utt] = utt
    utt2vocals[utt] = vocals == "Y"
  return utt2spk, utt2vocals

def prepare_music(root_dir, use_vocals):
  utt2vocals = {}
  utt2spk = {}
  utt2wav = {}
  music_dir = os.path.join(root_dir, "music")
  print str(music_dir)
  for root, dirs, files in os.walk(music_dir):
    for file in files:
      file_path = os.path.join(root, file)
      if file.endswith(".wav"):
        utt = str(file).replace(".wav", "")
        utt2wav[utt] = file_path
      elif str(file) == "ANNOTATIONS":
        utt2spk_part, utt2vocals_part = process_music_annotations(file_path)
        utt2spk.update(utt2spk_part)
        utt2vocals.update(utt2vocals_part)
  utt2spk_str = ""
  utt2wav_str = ""
  for utt in utt2vocals:
    if use_vocals or not utt2vocals[utt]:
      utt2spk_str = utt2spk_str + utt + " " + utt2spk[utt] + "\n"
      utt2wav_str = utt2wav_str + utt + " " + utt2wav[utt] + "\n"
  return utt2spk_str, utt2wav_str

def prepare_speech(root_dir):
  utt2spk = {}
  utt2wav = {}
  speech_dir = os.path.join(root_dir, "speech")
  for root, dirs, files in os.walk(speech_dir):
    for file in files:
      file_path = os.path.join(root, file)
      if file.endswith(".wav"):
        utt = str(file).replace(".wav", "")
        utt2wav[utt] = file_path
        utt2spk[utt] = utt
  utt2spk_str = ""
  utt2wav_str = ""
  for utt in utt2spk:
    utt2spk_str = utt2spk_str + utt + " " + utt2spk[utt] + "\n"
    utt2wav_str = utt2wav_str + utt + " " + utt2wav[utt] + "\n"
  return utt2spk_str, utt2wav_str

def prepare_noise(root_dir):
  utt2spk = {}
  utt2wav = {}
  speech_dir = os.path.join(root_dir, "noise")
  for root, dirs, files in os.walk(speech_dir):
    for file in files:
      file_path = os.path.join(root, file)
      if file.endswith(".wav"):
        utt = str(file).replace(".wav", "")
        utt2wav[utt] = file_path
        utt2spk[utt] = utt
  utt2spk_str = ""
  utt2wav_str = ""
  for utt in utt2spk:
    utt2spk_str = utt2spk_str + utt + " " + utt2spk[utt] + "\n"
    utt2wav_str = utt2wav_str + utt + " " + utt2wav[utt] + "\n"
  return utt2spk_str, utt2wav_str

def main():
  in_dir = sys.argv[1]
  out_dir = sys.argv[2]
  use_vocals = sys.argv[3] == "Y"
  utt2spk_music, utt2wav_music = prepare_music(in_dir, use_vocals)
  utt2spk_speech, utt2wav_speech = prepare_speech(in_dir)
  utt2spk_noise, utt2wav_noise = prepare_noise(in_dir)
  utt2spk = utt2spk_speech + utt2spk_music + utt2spk_noise
  utt2wav = utt2wav_speech + utt2wav_music + utt2wav_noise
  wav_fi = open(os.path.join(out_dir, "wav.scp"), 'w')
  wav_fi.write(utt2wav)
  utt2spk_fi = open(os.path.join(out_dir, "utt2spk"), 'w')
  utt2spk_fi.write(utt2spk)


if __name__=="__main__":
  main()
