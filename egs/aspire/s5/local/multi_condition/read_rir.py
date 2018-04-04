#!/usr/bin/env python
# Copyright 2014  Johns Hopkins University (Authors: Vijayaditya Peddinti).  Apache 2.0.
# script to read rir files from air database

import sys, numpy as np, argparse, scipy.signal as signal, os.path, glob, scipy.io, scipy.io.wavfile

def read_raw(input_filename, precision = np.float32):
  # assuming numpy return littleendian data
  file_handle = open(input_filename, 'rb')
  data = np.fromfile(file_handle, dtype = precision)
  #sys.stderr.write("Read file of length {0} and type {1}\n".format(len(data), precision))
  return data

def wav_write(file_handle, fs, data):
  if str(data.dtype) in set(['float64', 'float32']):
    data = (0.99 * data / np.max(np.abs(data))) * (2 ** 15)
    data = data.astype('int16', copy = False)
  elif str(data.dtype) == 'int16':
    pass
  else:
    raise Exception('Not implemented for '+str(data.dtype))
  scipy.io.wavfile.write(file_handle, fs, data)


def usage():
  return """This is a python script to read impulse responses stored in custom formats. It handles AIR database."""
   
if __name__ == "__main__":
  #sys.stderr.write(" ".join(sys.argv)+"\n")
  parser = argparse.ArgumentParser(usage())
  parser.add_argument('--output-sampling-rate', type = int, default = 8000,  help = 'sampling rate of the output')
  parser.add_argument('type', type = str, default = None,  help = 'database type', choices = ['air'])
  parser.add_argument('input', type = str, default = None,  help = 'directory containing the multi-channel data for a particular recording, or file name or file-regex-pattern')
  parser.add_argument('output_filename', type = str, default = None,  help = 'output filename (if "-" then output is written to output pipe)')
  params = parser.parse_args()

  if params.output_filename == "-":
    output = sys.stdout
  else:
    output = open(params.output_filename, 'wb')

  if params.type == 'air':
    files = glob.glob(params.input)
    # there are just two files which vary in the channel id (0,1)
    assert(len(files)==2)
    sr = -1
    data = []
    for file in files:
      mat_data = scipy.io.loadmat(file) 
      data.append(mat_data['h_air'][0,:])
      sr = mat_data['air_info']['fs'][0][0][0][0]
    data = np.array(data)
    data = data.transpose()
    assert(data.shape[1] == 2)
    if params.output_sampling_rate != sr:
      data = signal.resample(data,  int(params.output_sampling_rate * float(data.shape[0]) / sr), axis = 0)
  wav_write(output, params.output_sampling_rate, data)
