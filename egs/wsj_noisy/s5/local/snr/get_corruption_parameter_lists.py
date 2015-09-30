#!/usr/bin/env python
# Copyright 2014  Johns Hopkins University (Authors: Vijayaditya Peddinti).
#           2015  Vimal Manohar
# Apache 2.0.
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

# Return non-empty lines from a list of lines
def return_nonempty_lines(lines):
  new_lines = []
  for line in lines:
    if len(line.strip()) > 0:
      new_lines.append(line.strip())
  return new_lines

def exists_wavfile(file_name):
  return os.path.isfile(file_name)
  try:
    scipy.io.wavfile.read(file_name)
    return True
  except IOError:
    return False

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--snrs', type=str, default = '20:10:0',
                      help='snrs to be used for corruption')
  parser.add_argument('--num-files-per-job', type=int, default = None,
                      help='number of commands for corruption to be stored in each file -- This is the number of parallel jobs that will be run')
  parser.add_argument('--check-output-exists', type = str, default = 'True',
                      help = 'process file only if output file does not exist', choices = ['True', 'true', 'False', 'false'])
  parser.add_argument('--random-seed', type = int, default = 0,
                      help = 'seed to be used in the randomization of corruption')
  parser.add_argument('--clean-wav-scp-file', type=str, help='list file to write the clean output file before adding noise to create corrupted file')
  parser.add_argument('--noise-wav-scp-file', type=str, help='list file to write the noise that is added to create the corrupted output file')
  parser.add_argument('wav_scp_file', type=str, help='wav.scp file to corrupt')
  parser.add_argument('output_wav_scp_file', type=str, help='list file to write corrupted output')
  parser.add_argument('impulses_noises_dir', type=str, help='directory with impulses and noises and info directory (e.g. created by local/multicondition/prep_rirs.sh)')
  parser.add_argument('output_command_file', type=str, help='file to output the corruption commands')
  params = parser.parse_args()

  add_noise = True
  snr_string_parts = params.snrs.split(':')
  if (len(snr_string_parts) == 1) and snr_string_parts[0] == "inf":
    add_noise = False
  snrs = list_cyclic_iterator(snr_string_parts, random_seed = params.random_seed)

  signal_db_string_parts = params.signal_dbs.split(':')
  signal_dbs = list_cyclic_iterator(signal_db_string_parts, random_seed = params.random_seed)

  if params.check_output_exists.lower() == 'true':
    params.check_output_exists = True
  else:
    params.check_output_exists = False

  wav_files = return_nonempty_lines(open(params.wav_scp_file, 'r').readlines())
  wav_out_files = return_nonempty_lines(open(params.output_wav_scp_file, 'r').readlines())
  assert(len(wav_files) == len(wav_out_files))
  if params.clean_wav_scp_file is not None:
    clean_wav_out_files = return_nonempty_lines(open(params.clean_wav_scp_file, 'r').readlines())
    assert(len(wav_files) == len(clean_wav_out_files))
  if params.noise_wav_scp_file is not None:
    noise_wav_out_files = return_nonempty_lines(open(params.noise_wav_scp_file, 'r').readlines())
    assert(len(wav_files) == len(noise_wav_out_files))
  impulses = list_cyclic_iterator(return_nonempty_lines(open(params.impulses_noises_dir+'/info/impulse_files').readlines()), random_seed = params.random_seed)    # This list could be empty
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

  if params.num_files_per_job is None:
    lines_per_file = len(wav_files)
  else:
    lines_per_file = params.num_files_per_job
  num_parts = int(math.ceil(len(wav_files)/ float(lines_per_file))) # The number of parallel jobs
  indices_per_file = map(lambda x: xrange(lines_per_file * (x-1), lines_per_file * x), range(1, num_parts))
  indices_per_file.append(xrange(lines_per_file * (num_parts-1), len(wav_files)))

  part_counter = 1
  commands_file_base, ext = os.path.splitext(params.output_command_file)
  for indices in indices_per_file:
    command_list = []
    for i in indices:
      wav_file = " ".join(wav_files[i].split()[1:])   # Can be a pipe input
      output_wav_file = wav_out_files[i]              # An actual wave file
      clean_wav_file = ''
      noise_wav_file = ''
      if params.clean_wav_scp_file is not None:
        clean_wav_file = ''.join(['--out-clean-file ', clean_wav_out_files[i], ' '])
      if params.noise_wav_scp_file is not None:
        noise_wav_file = ''.join(['--out-noise-file ', noise_wav_out_files[i], ' '])
      impulse_file = impulses.next()                  # Can be None
      noise_file = ''
      snr = ''
      signal_db = ''
      found_impulse = (impulse_file is not None)
      found_noise = False
      if add_noise:
        for i in xrange(len(impulse_noise_index)):
          if impulse_file is None and not impulse_noise_index[i][0]:
            noise_file = impulse_noise_index[i][1].next()
            snr = snrs.next()
            signal_db = signal_dbs.next()
            assert(len(wav_file.strip()) > 0)
            assert(len(noise_file.strip()) > 0)
            assert(len(snr.strip()) > 0)
            assert(len(signal_db.strip() > 0)
            assert(len(output_wav_file.strip()) > 0)
            command_list.append("{0} --noise-file {1} --snr-db {2} --signal-db {3} {4}{5}- {6} \n".format(wav_file, noise_file, snr, signal_db, clean_wav_file, noise_wav_file, output_wav_file))
            found_noise = True
            break
          if impulse_file in impulse_noise_index[i][0]:
            noise_file = impulse_noise_index[i][1].next()
            snr = snrs.next()
            signal_db = signal_dbs.next()
            assert(len(wav_file.strip()) > 0)
            assert(len(impulse_file.strip()) > 0)
            assert(len(noise_file.strip()) > 0)
            assert(len(snr.strip()) > 0)
            assert(len(signal_db.strip() > 0)
            assert(len(output_wav_file.strip()) > 0)
            command_list.append("{0} --rir-file {1} --noise-file {2} --snr-db {3} --signal-db {4} {5}{6}- {7} \n".format(wav_file, impulse_file, noise_file, snr, signal_db, clean_wav_file, noise_wav_file, output_wav_file))
            found_impulse = True
            found_noise = True
            break
      if not found_noise:
        assert (found_impulse)
        assert(len(wav_file.strip()) > 0)
        assert(len(impulse_file.strip()) > 0)
        assert(len(output_wav_file.strip()) > 0)
        command_list.append("{0} --rir-file {1} {2}{3}- {4} \n".format(wav_file, impulse_file, clean_wav_file, noise_wav_file, output_wav_file))
      if params.check_output_exists and exists_wavfile(output_wav_file):
        # we perform the check at this point to ensure replication of (wavfile, impulse, noise, snr) tuples across runs.
        command_list.pop()
    file_handle = open("{0}.{1}{2}".format(commands_file_base, part_counter, ext), 'w')
    part_counter += 1
    file_handle.write("".join(command_list))
    file_handle.close()
  print num_parts
