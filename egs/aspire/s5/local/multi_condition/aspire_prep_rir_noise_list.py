#!/usr/bin/env python
# Copyright 2016  Tom Ko
# Apache 2.0
# script to generate rir_list and noise_list in aspire

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import argparse, glob, math, os, sys


def GetArgs():
    parser = argparse.ArgumentParser(description="Prepare rir_list and noise_list for Aspire. "
                                     "For information on the the format of data/impulses_noises "
                                     "look at the script aspire/s5/local/multi_condition/prepare_impulses_noises.sh "
                                     "This script assumes the data/impulses_noises directory has been prepared "
                                     "by the command aspire/s5/local/multi_condition/prepare_impulses_noises.sh "
                                     "--db-string \"'air' 'rvb2014' 'rwcp'\". It does not work for other directories. "
                                     "Usage: aspire_prep_rir_noise_list.py [options...] <in-data-dir> <out-data-dir> "
                                     "E.g. aspire_prep_rir_noise_list.py "
                                     "data/impulses_noises data/impulses_noises/info",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", help="Input data directory")
    parser.add_argument("output_dir", help="Output data directory")
    print(' '.join(sys.argv))
    args = parser.parse_args()

    return args


# This function generates the rir_list file for the real RIRs being in ASpIRE experiments.
# It assumes the availability of data/impulses_noises directory prepared by local/multi_condition/prepare_impulses_noises.sh
def GenerateRirListFile(input_dir, output_dir):
  rir_list_file = open(output_dir + "/rir_list", 'w')
  rir_id = 1
  for db in ["RVB2014", "RWCP", "air"]:
    rir_files = glob.glob(input_dir + "/{0}_*.wav".format(db))
    rir_files.sort()
    for rir in rir_files:
      filename = rir.split('/')[-1]
      if "noise" not in filename:
        parts = filename.split('_')
        db_name = parts[0]
        type_num = parts[1]
        if db == "RVB2014":
          noise_pattern = parts[3]
        elif db == "RWCP" and len(parts) == 4:
          noise_pattern = parts[3]
        else:
          noise_pattern = '_'.join(parts[3:len(parts)-1])

        # We use the string as the room id
        room_id = db_name + "_" + noise_pattern
        rir_list_file.write('--rir-id {0} --room-id {1} {2}\n'.format(str(rir_id).zfill(5), room_id, rir))
        rir_id += 1
  rir_list_file.close()


# This function generate the noise_list file from the aspire noise-rir pair 
def GenerateNoiseListFile(input_dir, output_dir):
  noise_list_file = open(output_dir + "/noise_list", 'w')
  noise_files = glob.glob(input_dir + "/*_type*_noise*.wav")
  noise_files.sort()
  noise_id = 1
  for noise_file in noise_files:
    parts = noise_file.split('/')[-1].split('_')
    db_name = parts[0]
    type_num = parts[1]
    noise_pattern = '_'.join(parts[3:len(parts)-1])
    noise_line = "--noise-id {0} --noise-type isotropic ".format(str(noise_id).zfill(5))
    room_id = db_name + "_" + noise_pattern
    noise_line += "--room-linkage {0} ".format(room_id)
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

