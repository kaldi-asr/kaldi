#!/usr/bin/env python

# Copyright Johns Hopkins University (Author: Vijayaditya Peddinti) 2015.  Apache 2.0.
# This script checks the ctm for missing recordings and places a dummy id.
# This is necessary for compliance with sclite scoring scripts

import argparse, sys

def fill_ctm(input_ctm_file, output_ctm_file, recording_names):
  recording_index = 0
  with open(input_ctm_file, "r") as infile, open(output_ctm_file, "w") as outfile:
    for line in infile:
      if line.split()[0] == recording_names[recording_index]:
        outfile.write(line)
      else:
        processed_line = False
        recording_index += 1
        while not processed_line:
          if recording_index >= len(recording_names):
            raise Exception("There is a mismatch between the recording_names_file and the ctm file. There are recordings in ctm file which are not present in the recording file.")
          if line.split()[0] == recording_names[recording_index]:
            outfile.write(line)
            processed_line = True
          else:
            # there is a missing recording
            outfile.write("{0} 1 0.00  0.01 NOTHINGWASDECODEDHERE\n".format(recording_names[recording_index]))
            recording_index += 1
    infile.close()
    outfile.close()



if __name__ == "__main__":
  usage = """ Python script to check the ctm file for missing recordings
  and provide a single line default output for the missing recordings. It assumes
  that the ctm file has recordings in the same order as the wav.scp file"""


  sys.stderr.write(str(" ".join(sys.argv)))
  parser = argparse.ArgumentParser(usage)
  parser.add_argument('input_ctm_file', type=str, help='ctm file for the recordings')
  parser.add_argument('output_ctm_file', type=str, help='ctm file for the recordings')
  parser.add_argument('recording_name_file', type=str, help='file with names of the recordings')

  params = parser.parse_args()

  try:
    file_names = map(lambda x: x.strip(), open("{0}".format(params.recording_name_file)).readlines())
  except IOError:
    raise Exception("Expected to find {0}".format(params.recording_name_file))

  fill_ctm(params.input_ctm_file, params.output_ctm_file, file_names)
