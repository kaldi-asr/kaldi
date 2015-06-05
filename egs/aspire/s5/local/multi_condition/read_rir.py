#!/usr/bin/env python
# Copyright 2014  Johns Hopkins University (Authors: Vijayaditya Peddinti).  Apache 2.0.

# script to read rir files from rwcp/air/reverb2014 databases

import sys, numpy as np, argparse, scipy.signal as signal, os.path, glob, scipy.io, scipy.io.wavfile

def read_raw(input_filename, precision = np.float32):
  # assuming numpy return littleendian data
  file_handle = open(input_filename, 'rb')
  data = np.fromfile(file_handle, dtype = precision)
  #sys.stderr.write("Read file of length {0} and type {1}\n".format(len(data), precision))
  return data

def usage():
  return """This is a python script to read various types of impulse responses stored in custom formats. It handles RWCP."""
   
if __name__ == "__main__":
  #sys.stderr.write(" ".join(sys.argv)+"\n")
  parser = argparse.ArgumentParser(usage())
  parser.add_argument('--output-sampling-rate', type = int, default = 8000,  help = 'sampling rate of the output')
  parser.add_argument('type', type = str, default = None,  help = 'database type', choices = ['rwcp/rir', 'rwcp/noise', 'air', 'reverb2014'])
  parser.add_argument('input', type = str, default = None,  help = 'directory containing the multi-channel data for a particular recording, or file name or file-regex-pattern')
  parser.add_argument('output_filename', type = str, default = None,  help = 'output filename (if "-" then output is written to output pipe)')
  params = parser.parse_args()

  if params.output_filename == "-":
    output = sys.stdout
  else:
    output = open(params.output_filename, 'wb')

  if params.type in set(['rwcp/rir', 'rwcp/noise']):
    if params.type == 'rwcp/rir':
      precision = np.float32
    elif params.type == 'rwcp/noise':
      precision = np.int16
    sr = 48000
    if (os.path.isdir(params.input)):
      channel_files = glob.glob(params.input+'/*')
      channel_file_basenames = map(lambda x: os.path.basename(x), channel_files)
      # check if all the files are present
      file_name = set(map(lambda x: x.split(".")[0], channel_files))
      assert(len(file_name) == 1)
      extensions = map(lambda x: int(x.split(".")[1]), channel_files)
      extensions.sort()
      assert(extensions == range(min(extensions), max(extensions)+1))
      base_name = ".".join(channel_files[0].split(".")[:-1])
      data = []
      for ext in extensions:
        data.append(read_raw("{0}.{1}".format(base_name, ext), precision = precision))
      data = np.array(data)
      data = data.transpose()
    else:
      data= []
      data.append(read_raw(params.input, precision = precision))
      data = np.array(data)
      data = data.transpose()
      assert(data.shape[1] == 1)
    if params.output_sampling_rate != sr:
      data = signal.resample(data,  params.output_sampling_rate * float(data.shape[0]) / sr, axis = 0)
  elif params.type == 'air':
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
      data = signal.resample(data,  params.output_sampling_rate * float(data.shape[0]) / sr, axis = 0)
  elif params.type == 'reverb2014':
    wav_data = scipy.io.wavfile.read(params.input)
    sr = wav_data[0]
    data = wav_data[1]
    if params.output_sampling_rate != sr:
      data = signal.resample(data,  params.output_sampling_rate * float(data.shape[0]) / sr, axis = 0)
  scipy.io.wavfile.write(output, params.output_sampling_rate, data)
