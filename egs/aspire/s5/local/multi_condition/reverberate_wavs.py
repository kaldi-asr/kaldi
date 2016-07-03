#!/usr/bin/env python
# Copyright 2014  Johns Hopkins University (Authors: Vijayaditya Peddinti).  Apache 2.0.
#           2015  Tom Ko
# script to generate multicondition training data / dev data / test data
import argparse, glob, math, os, random, scipy.io.wavfile, sys

class list_cyclic_iterator:
  def __init__(self, list, random_seed = 0):
    self.list_index = 0
    self.list = list
    random.seed(random_seed)
    random.shuffle(self.list)

  def next(self):
    item = self.list[self.list_index]
    self.list_index = (self.list_index + 1) % len(self.list)
    return item

def return_nonempty_lines(lines):
  new_lines = []
  for line in lines:
    if len(line.strip()) > 0:
      new_lines.append(line.strip())

  return new_lines

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--snrs', type=str, default = '20:10:0', help='snrs to be used for corruption')
  parser.add_argument('--check-output-exists', type = str, default = 'True', help = 'process file only if output file does not exist', choices = ['True', 'true', 'False', 'false'])
  parser.add_argument('--random-seed', type = int, default = 0, help = 'seed to be used in the randomization of impulses')
  parser.add_argument('wav_file_list', type=str, help='wav.scp file to corrupt')
  parser.add_argument('output_wav_file_list', type=str, help='wav.scp file to write corrupted output')
  parser.add_argument('impulses_noises_dir', type=str, help='directory with impulses and noises and info directory (created by local/prep_rirs.sh)')
  parser.add_argument('output_command_file', type=str, help='file to output the corruption commands')
  params = parser.parse_args() 
  
  add_noise = True
  snr_string_parts = params.snrs.split(':')
  if (len(snr_string_parts) == 1) and snr_string_parts[0] == "inf":
    add_noise = False
  snrs = list_cyclic_iterator(params.snrs.split(':'))
  if params.check_output_exists.lower == 'True':
    params.check_output_exists = True
  else:
    params.check_output_exists = False

  wav_files = return_nonempty_lines(open(params.wav_file_list, 'r').readlines())
  wav_out_files = return_nonempty_lines(open(params.output_wav_file_list, 'r').readlines())
  assert(len(wav_files) == len(wav_out_files))
  impulses = list_cyclic_iterator(return_nonempty_lines(open(params.impulses_noises_dir+'/info/impulse_files').readlines()), random_seed = params.random_seed)
  noises_impulses_files = glob.glob(params.impulses_noises_dir+'/info/noise_impulse_*')
  impulse_noise_index = []
  for file in noises_impulses_files:
    noises_list = []
    impulses_set = set([])
    for line in return_nonempty_lines(open(file).readlines()):
      line = line.strip()
      if len(line) == 0 or line[0] == '#':
        continue
      parts = line.split('=')
      if parts[0].strip() == 'noise_files':
        noises_list = list_cyclic_iterator(parts[1].split())
      elif parts[0].strip() == 'impulse_files':
        impulses_set = set(parts[1].split())
      else:
        raise Exception('Unknown format of ' + file)
      impulse_noise_index.append([impulses_set, noises_list])

  command_list = []
  for i in range(len(wav_files)):
    wav_file = " ".join(wav_files[i].split()[1:])
    output_wav_file = wav_out_files[i]
    impulse_file = impulses.next()
    noise_file = ''
    snr = ''
    found_impulse = False
    if add_noise:
      for i in xrange(len(impulse_noise_index)):
        if impulse_file in impulse_noise_index[i][0]:
          noise_file = impulse_noise_index[i][1].next()
          snr = snrs.next()
          assert(len(wav_file.strip()) > 0)
          assert(len(impulse_file.strip()) > 0)
          assert(len(noise_file.strip()) > 0)
          assert(len(snr.strip()) > 0)
          assert(len(output_wav_file.strip()) > 0)
          command_list.append("{4} {0} wav-reverberate --noise-file={2} --snr-db={3} - {1} - |\n".format(wav_file, impulse_file, noise_file, snr, output_wav_file))
          found_impulse = True
          break
    if not found_impulse:
      assert(len(wav_file.strip()) > 0)
      assert(len(impulse_file.strip()) > 0)
      assert(len(output_wav_file.strip()) > 0)
      command_list.append("{2} {0} wav-reverberate - {1} - |\n".format(wav_file, impulse_file, output_wav_file))
  file_handle = open(params.output_command_file, 'w')
  file_handle.write("".join(command_list))
  file_handle.close()
