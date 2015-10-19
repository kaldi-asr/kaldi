#!/usr/bin/env python
# Copyright 2014  Johns Hopkins University (Authors: Vijayaditya Peddinti).  Apache 2.0.
#           2015  Tom Ko
#           2015  Vimal Manohar
# script to generate multicondition training data / dev data / test data
import argparse, glob, math, os, random, scipy.io.wavfile, sys

class list_cyclic_iterator:
  def __init__(self, list, random_seed = 0):
    self.list_index = 0
    self.list = list
    random.seed(random_seed)
    random.shuffle(self.list)

  def next(self):
    if (len(self.list) == 0):
      return None
    item = self.list[self.list_index]
    self.list_index = (self.list_index + 1) % len(self.list)
    return item

  def add_to_list(self, a):
    self.list.insert(random.randrange(len(self.list)+1), a)

def return_nonempty_lines(lines):
  new_lines = []
  for line in lines:
    if len(line.strip()) > 0:
      new_lines.append(line.strip())
  return new_lines

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--snrs', type=str, default = '20:10:0', help='snrs to be used for corruption')
  parser.add_argument('--signal-dbs', type=str, default = '5:2:0:-2:-5:-10:-20:-30:-40:-50:-60',
                      help='desired signal dbs')
  parser.add_argument('--random-seed', type = int, default = 0, help = 'seed to be used in the randomization of impulses')
  parser.add_argument('--output-clean-wav-file-list', type=str, help='file to write clean output')
  parser.add_argument('--output-noise-wav-file-list', type=str, help='file to write noise output')
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

  signal_dbs= list_cyclic_iterator(params.signal_dbs.split(':'))

  wav_files = return_nonempty_lines(open(params.wav_file_list, 'r').readlines())
  wav_out_files = return_nonempty_lines(open(params.output_wav_file_list, 'r').readlines())
  assert(len(wav_files) == len(wav_out_files))

  if params.output_clean_wav_file_list is not None:
    clean_wav_out_files = return_nonempty_lines(open(params.output_clean_wav_file_list, 'r').readlines())
    assert(len(wav_files) == len(clean_wav_out_files))
  if params.output_noise_wav_file_list is not None:
    noise_wav_out_files = return_nonempty_lines(open(params.output_noise_wav_file_list, 'r').readlines())
    assert(len(wav_files) == len(noise_wav_out_files))

  impulses = list_cyclic_iterator(return_nonempty_lines(open(params.impulses_noises_dir+'/info/impulse_files').readlines()), random_seed = params.random_seed)    # This list could be empty
  impulses.add_to_list(None)
  noises = list_cyclic_iterator(return_nonempty_lines(open(params.impulses_noises_dir+'/info/noise_files').readlines()), random_seed = params.random_seed)

  #noises_impulses_files = glob.glob(params.impulses_noises_dir+'/info/noise_impulse_*')
  #impulse_noise_index = []

  #for file in noises_impulses_files:
  #  noises_list = []
  #  impulses_set = set([])
  #  for line in return_nonempty_lines(open(file).readlines()):
  #    line = line.strip()
  #    if len(line) == 0 or line[0] == '#':
  #      continue
  #    parts = line.split('=')
  #    if parts[0].strip() == 'noise_files':
  #      noises_list = list_cyclic_iterator(parts[1].split())
  #    elif parts[0].strip() == 'impulse_files':
  #      impulses_set = set(parts[1].split())
  #    else:
  #      raise Exception('Unknown format of ' + file)
  #  impulse_noise_index.append([impulses_set, noises_list])

  command_list = []
  for i in range(len(wav_files)):
    wav_file = " ".join(wav_files[i].split()[1:])
    splits = wav_out_files[i].split()
    output_filename = splits[0]
    output_wav_file = " ".join(splits[1:])
    impulse_file = impulses.next()
    noise_file = noises.next()
    signal_db = signal_dbs.next()
    snr = snrs.next()

    assert(len(wav_file.strip()) > 0)
    assert(impulse_file is None or len(impulse_file.strip()) > 0)
    assert(len(noise_file.strip()) > 0)
    assert(len(snr.strip()) > 0)
    assert(len(output_wav_file.strip()) > 0)

    if impulse_file is None:
      impulse_file = ''

    volume_opts = ""
    if signal_db is not None:
      assert(len(signal_db.strip()) > 0)
      volume_opts = "--volume=-1 --signal-db={0} --normalize-by-amplitude=true".format(signal_db)

    if params.output_clean_wav_file_list is not None:
      splits = clean_wav_out_files[i].split()
      assert(output_filename == splits[0])
      output_clean_wav_opts = "--output-clean-file={0}".format(" ".join(splits[1:]))
    if params.output_noise_wav_file_list is not None:
      splits = noise_wav_out_files[i].split()
      assert(output_filename == splits[0])
      output_noise_wav_opts = "--output-noise-file={0}".format(" ".join(splits[1:]))
    command_list.append("{8} {0} wav-reverberate --noise-file={2} --snr-db={3} {4} {5} {6} - {1} {7}\n".format(wav_file, impulse_file, noise_file, snr, volume_opts, output_clean_wav_opts, output_noise_wav_opts, output_wav_file, output_filename))

  file_handle = open(params.output_command_file, 'w')
  file_handle.write("".join(command_list))
  file_handle.close()

