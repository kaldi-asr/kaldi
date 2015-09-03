#! /usr/bin/env python

import argparse, sys
from argparse import ArgumentParser
import re

def main():
  parser = ArgumentParser(description='Convert kaldi data directory to uem dat files',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--verbose', type=int, \
      dest='verbose', default=0, \
      help='Give higher verbose for more logging')
  parser.add_argument('--get-text', action='store_true', \
      help='Get text in dat file')
  parser.add_argument('--prefix', type=str, \
      help='Add db file name as db-<prefix>-{utt/spk}.dat')
  parser.add_argument('kaldi_dir', \
      help='Kaldi data directory')
  parser.add_argument('output_dir', \
      help='Directory to store uem dat files')
  parser.usage=':'.join(parser.format_usage().split(':')[1:]) \
      + 'e.g. :  %(prog)s --prefix 203-lao-v0 data/dev10h.seg CMU_db'
  options = parser.parse_args()

  if options.get_text:
    try:
      text_file = open(options.kaldi_dir+'/text', 'r')
    except IOError as e:
      repr(e)
      sys.stderr.write("%s: No such file %s\n" % (sys.argv[0], options.kaldi_dir+'/text'))
      sys.exit(1)

  try:
    segments_file = open(options.kaldi_dir+'/segments', 'r')
  except IOError as e:
    repr(e)
    sys.stderr.write("%s: No such file %s\n" % (sys.argv[0], options.kaldi_dir+'/segments'))
    sys.exit(1)

  try:
    scp_file = open(options.kaldi_dir+'/wav.scp', 'r')
  except IOError as e:
    repr(e)
    sys.stderr.write("%s: No such file %s\n" % (sys.argv[0], options.kaldi_dir+'/wav.scp'))
    sys.exit(1)

  reco2file_map = {}
  for line in scp_file.readlines():
    splits = line.strip().split()
    m = re.search(r".*/(?P<file_name>[0-9A-Za-z_]*\.(sph|wav)).*", line)
    if not m:
      sys.stderr.write("%s does not contain a valid speech file (.wav or .sph)\n" % line.strip())
      sys.exit(1)
    reco2file_map[splits[0]] = m.group('file_name')
  # End for

  spk2utt_map = {}

  if options.prefix == None:
    prefix = options.kaldi_dir.split('/')[-1].split('.')[0]
  else:
    prefix = options.prefix

  try:
    utt_dat = open(options.output_dir+'/db-'+prefix+'-utt.dat', 'w')
    spk_dat = open(options.output_dir+'/db-'+prefix+'-spk.dat', 'w')
  except IOError as e:
    repr(e)
    sys.stderr.write("%s: Could not write dat files in %s\n" % (sys.argv[0], options.output_dir))
    sys.exit(1)

  for line in segments_file.readlines():
    utt_id, file_id, start, end = line.strip().split()

    if (options.get_text):
      splits = text_file.readline().split()
      while splits[0] < utt_id:
        splits = text_file.readline().split()
      text = ' '.join(splits[1:])
    else:
      text = ""

    utt_dat.write("{UTTID %s} {UTT %s} {SPK %s} {FROM %s} {TO %s} {TEXT %s}\n" % (utt_id, utt_id, file_id, start, end, text))
    spk2utt_map.setdefault(file_id, [])
    spk2utt_map[file_id].append(utt_id)

  for spk, utts in spk2utt_map.items():
    try:
      spk_dat.write("{SEGS %s} {ADC %s} {CONV %s.wav} {CHANNEL 1} {DUR }\n" % (' '.join(utts), reco2file_map[spk], spk))
    except KeyError as e:
      repr(e)
      sys.stderr.write("%s: Error in getting file for %s\n" % (sys.argv[0], spk))
      sys.exit(1)
  # End for

  segments_file.close()
  utt_dat.close()
  spk_dat.close()

if __name__ == '__main__':
  main()
