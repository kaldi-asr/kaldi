#!/usr/bin/env python
# Copyright 2016  Tom Ko
# Apache 2.0
# script to generate rir_list and noise_list in aspire

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import argparse, glob, math, os, sys


def GetArgs():
    parser = argparse.ArgumentParser(description="Prepare rir_list and noise_list for Aspire  "
                                                 "Usage: reverberate_data_dir.py [options...] <in-data-dir> <out-data-dir> "
                                                 "E.g. reverberate_data_dir.py "
                                                 "data/impulses_noises data/impulses_noises/info",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", help="Input data directory")
    parser.add_argument("output_dir", help="Output data directory")
    print(' '.join(sys.argv))
    args = parser.parse_args()

    return args


# This function generate the rir_list file for the aspire real RIR
def GenerateRirListFile(input_dir, output_dir):
  rir_list_file = open(output_dir + "/rir_list", 'w')
  rir_id = 1
  room_id = 1
  for db in ["RVB2014", "RWCP", "air"]:
    rir_files = glob.glob(input_dir + "/{0}_*.wav".format(db))
    for rir in rir_files:
      filename = rir.split('/')[-1]
      if "noise" not in filename:
        rir_list_file.write('--rir-id {0} --room-id {1} {2}\n'.format(str(rir_id).zfill(5), str(room_id).zfill(3), rir))
        rir_id += 1
    room_id += 1
  rir_list_file.close()


# This function generate the noise_list file from the aspire noise-rir pair 
def GenerateNoiseListFile(input_dir, output_dir):
  noise_list_file = open(output_dir + "/noise_list", 'w')
  noise_files = glob.glob(input_dir + "/*_type*_noise*.wav")
  noise_id = 1
  for noise_file in noise_files:
    parts = noise_file.split('/')[-1].split('_')
    db_name = parts[0]
    type_num = parts[1]
    noise_pattern = '_'.join(parts[3:len(parts)-1])
    if db_name == "RWCP":
      type_num = "type*"
    matched_rir_files = glob.glob(input_dir + "/{0}_{1}_rir_{2}*.wav".format(db_name, type_num, noise_pattern))
    noise_line = "--noise-id {0} --noise-type isotropic ".format(str(noise_id).zfill(5))
    for rir in matched_rir_files:
      noise_line += "--rir-linkage {0} ".format(rir)
    noise_line += "{0}".format(noise_file)
    noise_list_file.write("{0}\n".format(noise_line))
    noise_id += 1
  noise_list_file.close()


def Main():
  args = GetArgs()

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  # generating the rir_list file for the new steps/data/reverberate_data_dir.py
  GenerateRirListFile(args.input_dir, args.output_dir)

  # generating the noise_list file for the new steps/data/reverberate_data_dir.py
  GenerateNoiseListFile(args.input_dir, args.output_dir)


if __name__ == "__main__":
    Main()

