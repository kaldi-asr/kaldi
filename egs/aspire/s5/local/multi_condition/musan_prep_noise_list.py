#!/usr/bin/env python
# Copyright 2016  Tom Ko
# Apache 2.0
# script to generate noise_list for MUSAN corpus

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import argparse, glob, math, os, sys


def GetArgs():
    parser = argparse.ArgumentParser(description="Prepare noise_list from the MUSAN annotation file. "
                                     "It is used to distinguish the foreground/background noises. ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sampling-rate', type=int, default=8000, help='the target sampling rate for the noises')
    parser.add_argument("musan_dir", help="Input MUSAN directory")
    parser.add_argument("output_noise_list", help="Output file name for the noise list")
    print(' '.join(sys.argv))
    args = parser.parse_args()

    return args

# This function generate the noise_list file from the MUSAN annotation file
# The annotation file only record the filenames for background noises
def MusanAnnotationToNoiseList(musan_dir, output_noise_list, sampling_rate):
  background_list = map(lambda x: x.strip(), open(musan_dir + "/ANNOTATIONS", 'r'))

  noise_list_file = open(output_noise_list, 'w')
  noise_files = glob.glob(musan_dir + "/*.wav")
  noise_files.sort()
  for noise_file in noise_files:
    noise_id = noise_file.split('/')[-1].split('.')[0]
    noise_line = "--noise-id {0} --noise-type point-source ".format(noise_id)
    if noise_id in background_list:
      noise_line += "--bg-fg-type {0} ".format("background")
    else:
      noise_line += "--bg-fg-type {0} ".format("foreground")
    noise_line += "\"sox {0} -r {1} -t wav - |\"".format(noise_file, sampling_rate)
    noise_list_file.write("{0}\n".format(noise_line))
  noise_list_file.close()


def Main():
  args = GetArgs()
  MusanAnnotationToNoiseList(args.musan_dir, args.output_noise_list, args.sampling_rate)


if __name__ == "__main__":
    Main()

