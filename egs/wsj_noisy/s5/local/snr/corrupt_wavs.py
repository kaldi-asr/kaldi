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

  def next_few(self, num_items):
    if (len(self.list) == 0):
      return None
    a = []
    if (num_items > len(self.list)):
      num_items = len(self.list)
    for i in range(0, num_items):
      item = self.list[self.list_index]
      self.list_index = (self.list_index + 1) % len(self.list)
      a.append(item)
    return a

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
  parser.add_argument('--background-snrs', type=str, default = '20:10:0',
                      help='snrs to be used for corruption')
  parser.add_argument('--foreground-snrs', type=str, default = '20:10:0',
                      help='snrs to be used with foreground noises')
  parser.add_argument('--signal-dbs', type=str,
                      default = '5:2:0:-2:-5:-10:-20:-30:-40:-50:-60',
                      help='desired signal dbs')
  parser.add_argument('--foreground-prob', type=float, default = 0.7,
                      help = 'probability with which to add foreground noise '
                      'on non-pair impulse noise files')
  parser.add_argument('--foreground-prob-for-pair', type=float, default = 0.4,
                      help = 'probability with which to add foreground noise '
                      'on impulse-noise pairs')
  parser.add_argument('--random-seed', type = int, default = 0,
                      help = 'seed to be used in the randomization')
  parser.add_argument('--output-clean-wav-file-list', type=str,
                      help='file to write clean output')
  parser.add_argument('--output-noise-wav-file-list', type=str,
                      help='file to write noise output')
  parser.add_argument('wav_scp', type=str,
                      help='wav.scp file to corrupt')
  parser.add_argument('output_wav_scp', type=str,
                      help='wav.scp file to write corrupted output')
  parser.add_argument('impulses_noises_dir', type=str,
                      help='directory with impulses and noises and info '
                   'directory (created by local/snr/prepare_noise_impulses.sh)')
  parser.add_argument('output_command_file', type=str,
                      help='file to output the corruption commands')
  params = parser.parse_args()

  add_noise = True
  background_snr_string_parts = params.background_snrs.split(':')
  foreground_snr_string_parts = params.foreground_snrs.split(':')
  if (len(background_snr_string_parts) == 1 and
    background_snr_string_parts[0] == "inf" and
    len(foreground_snr_string_parts) == 1 and
    foreground_snr_string_parts[0] == "inf"):
    add_noise = False

  background_snrs = list_cyclic_iterator(background_snr_string_parts,
      random_seed = params.random_seed)
  foreground_snrs = list_cyclic_iterator(foreground_snr_string_parts,
      random_seed = params.random_seed)
  signal_dbs= list_cyclic_iterator(params.signal_dbs.split(':'),
      random_seed = params.random_seed)

  wav_files = return_nonempty_lines(open(params.wav_scp, 'r').readlines())
  wav_out_files = return_nonempty_lines(
                                open(params.output_wav_scp, 'r').readlines())
  assert(len(wav_files) == len(wav_out_files))

  if params.output_clean_wav_file_list is not None:
    clean_wav_out_files = return_nonempty_lines(
                      open(params.output_clean_wav_file_list, 'r').readlines())
    assert(len(wav_files) == len(clean_wav_out_files))
    # TODO: They must also be corresponding files. This must be checked
    # somewhere down the line.
  if params.output_noise_wav_file_list is not None:
    noise_wav_out_files = return_nonempty_lines(
                      open(params.output_noise_wav_file_list, 'r').readlines())
    assert(len(wav_files) == len(noise_wav_out_files))

  impulses = list_cyclic_iterator(return_nonempty_lines(
    open(params.impulses_noises_dir+'/info/impulse_files').readlines()),
    random_seed = params.random_seed)    # This list could be empty
  # Explicitly add a None so that you can create corrupted files without any reverberation
  impulses.add_to_list(None)

  background_noises = list_cyclic_iterator(return_nonempty_lines(
    open(params.impulses_noises_dir+'/info/background_noise_files').readlines()),
    random_seed = params.random_seed)
  # This must ideally not be empty because it will create infinities in SNR objective

  foreground_noises = list_cyclic_iterator(return_nonempty_lines(
    open(params.impulses_noises_dir+'/info/foreground_noise_files').readlines()),
    random_seed = params.random_seed)
  # This list could be empty too, which just means we won't be adding any foreground noise

  # noise-impulse pair files. If a background noise has a corresponding pair
  # then with a high probability, an rir paired with it will be selected.
  # Also there will be a low probability for adding foreground noise.
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
        noises_list = list_cyclic_iterator(parts[1].split(),
                                            random_seed = params.random_seed)
      elif parts[0].strip() == 'impulse_files':
        impulses_set = set(parts[1].split())
      else:
        raise Exception('Unknown format of ' + file)
      impulse_noise_index.append([impulses_set, noises_list])

  command_list = []
  for i in range(len(wav_files)):
    wav_file_splits = wav_files[i].split()
    wav_file = " ".join(wav_file_splits[1:])
    file_id = wav_file_splits[0]

    splits = wav_out_files[i].split()
    output_file_id = splits[0]
    output_wav_file = " ".join(splits[1:])

    # randomly select corruption parameters
    impulse_file = impulses.next()

    found_impulse_noise_pair = False
    if impulse_file is not None:
      for x in impulse_noise_index:
        if impulse_file in x[0]:
          found_impulse_noise_pair = True
          background_noise_file = x[1].next()
          foreground_prob = params.foreground_prob_for_pair

    if not found_impulse_noise_pair:
      foreground_prob = params.foreground_prob
      background_noise_file = background_noises.next()

    background_snr = background_snrs.next()
    signal_db = signal_dbs.next()

    assert(len(wav_file.strip()) > 0)
    assert(impulse_file is None or len(impulse_file.strip()) > 0)
    assert(len(background_noise_file.strip()) > 0)
    assert(len(background_snr.strip()) > 0)
    assert(len(output_wav_file.strip()) > 0)

    rir_opts = ''
    if impulse_file is not None:
      rir_opts = '--rir-file={0}'.format(impulse_file)

    background_noise_opts = '--background-noise-file={0} --background-snr-db={1}'.format(background_noise_file, background_snr)

    foreground_noise_opts = ''
    if random.uniform(0,1) >= foreground_prob:
      foreground_snr = foreground_snrs.next()
      foreground_noise_files = foreground_noises.next_few(10)
      if foreground_noise_files is not None:
        foreground_noise_opts = '--foreground-noise-files={0} --foreground-snr-db={1}'.format(":".join(foreground_noise_files), foreground_snr)

    volume_opts = ""
    if signal_db is not None:
      assert(len(signal_db.strip()) > 0)
      volume_opts = "--volume=-1 --signal-db={0} --normalize-by-amplitude=true".format(signal_db)

    if params.output_clean_wav_file_list is not None:
      splits = clean_wav_out_files[i].split()
      assert(output_file_id == splits[0])
      output_clean_wav_opts = "--output-clean-file={0}".format(" ".join(splits[1:]))
    if params.output_noise_wav_file_list is not None:
      splits = noise_wav_out_files[i].split()
      assert(output_file_id == splits[0])
      output_noise_wav_opts = "--output-noise-file={0}".format(" ".join(splits[1:]))

    # wav_file here is something like "cat <filename>.wav |"
    command = "{0} {1} corrupt-wav {2} {3} {4} {5} {6} {7} - {8}\n".format(output_file_id, wav_file, rir_opts, background_noise_opts, foreground_noise_opts, volume_opts, output_clean_wav_opts, output_noise_wav_opts, output_wav_file)
    command_list.append(command)

  file_handle = open(params.output_command_file, 'w')
  file_handle.write("".join(command_list))
  file_handle.close()

