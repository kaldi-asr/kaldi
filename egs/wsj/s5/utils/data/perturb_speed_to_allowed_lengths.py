#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (author: Hossein Hadian)
# Apache 2.0


""" This script perturbs speeds of utterances to force their lengths to some allowed
    lengths spaced by a factor
"""

import argparse
import os
import sys
import copy
import math

parser = argparse.ArgumentParser(description="""This script ...""")
parser.add_argument('factor', type=float, default=12,
                    help='spacing (in percentage) between allowed lengths.')
parser.add_argument('srcdir', type=str,
                    help='path to source data dir')
parser.add_argument('dir', type=str, help='output dir')
parser.add_argument('--range-factor', type=float, default=0.05,
                    help="""Percentage of durations not covered from each side of
                    duration histogram.""")
parser.add_argument('--no-speed-perturb',  action='store_true')

args = parser.parse_args()

### functions and classes ###

class Speaker:
  def __init__(self, path, sid):
    self.path = path
    self.name = os.path.basename(os.path.normpath(path))
    self.id = sid
    self.utterances = []
  def str_id(self):
    return "s" + zero_pad(str(self.id), 4)

class Utterance:
  def __init__(self, uid, wavefile, speaker, transcription, dur):
    self.wavefile = wavefile
    self.speaker = speaker
    self.transcription = transcription
    self.id = uid
    self.dur = float(dur)

  def to_kaldi_utt_str(self):
    return self.id + " " + self.transcription

  def to_kaldi_wave_str(self):
    return self.id + " " + self.wavefile


def read_kaldi_datadir(dir):
  utts = []
  wav_scp = read_kaldi_mapfile(os.path.join(dir, 'wav.scp'))
  text = read_kaldi_mapfile(os.path.join(dir, 'text'))
  utt2dur = read_kaldi_mapfile(os.path.join(dir, 'utt2dur'))
  utt2spk = read_kaldi_mapfile(os.path.join(dir, 'utt2spk'))
  for utt in wav_scp:
    if utt in text and utt in utt2dur and utt in utt2spk:
      utts += [Utterance(utt, wav_scp[utt], utt2spk[utt], text[utt], utt2dur[utt])]
    else:
      print('Incomplete data for utt {}'.format(utt))
  return utts


def read_kaldi_mapfile(path):
  m = {}
  with open(path, 'r') as f:
    for line in f:
      line = line.rstrip()
      sp_pos = line.find(' ')
      key = line[:sp_pos]
      val = line[sp_pos+1:]
      m[key] = val
  return m

def generate_kaldi_data_files(utterances, outdir):
  print "Exporting to ", outdir, "..."
  spks = {}

  f = open(os.path.join(outdir, 'text'), 'w')
  for utt in utterances:
    f.write(utt.to_kaldi_utt_str() + "\n")
  f.close()

  f = open(os.path.join(outdir, 'wav.scp'), 'w')
  for utt in utterances:
    f.write(utt.to_kaldi_wave_str() + "\n")
  f.close()

  f = open(os.path.join(outdir, 'utt2dur'), 'w')
  for utt in utterances:
    f.write(utt.id + " " + str(utt.dur) + "\n")
  f.close()

  f = open(os.path.join(outdir, 'utt2spk'), 'w')
  for utt in utterances:
    f.write(utt.id + " " + utt.speaker + "\n")
    if utt.speaker not in spks:
      spks[utt.speaker] = [utt.id]
    else:
      spks[utt.speaker] += [utt.id]
  f.close()

  f = open(os.path.join(outdir, 'spk2utt'), 'w')
  for s in spks:
    f.write(s + " ")
    for utt in spks[s]:
      f.write(utt + " ")
    f.write('\n')
  f.close()




### main ###

if not os.path.exists(args.dir):
  os.makedirs(args.dir)

# 0. load src dir
utts = read_kaldi_datadir(args.srcdir)

factor = 1.0 + float(args.factor)/100
# 1a. find start-dur and end-dur
## echo "Durs = [" >durs.m && cut -d' ' -f2 data/train_nodup_seg/utt2dur | tr '\n' ',' >>durs.m && echo " ];" >>durs.m
durs = []
for u in utts:
  durs += [u.dur]
durs.sort()
to_ignore_dur = 0
tot_dur = sum(durs)
for d in durs:
  to_ignore_dur += d
  if to_ignore_dur * 100.0 / tot_dur > args.range_factor:
    start_dur = d
    break
to_ignore_dur = 0
for d in reversed(durs):
  to_ignore_dur += d
  if to_ignore_dur * 100.0 / tot_dur > args.range_factor:
    end_dur = d
    break
print("Durations in the range [{},{}] will be covered. Coverage rate: {}%".format(start_dur, end_dur, 100.0-args.range_factor*2))
print("There will be {} unique allowed lengths for the utterances.".format(int(math.log(end_dur/start_dur)/math.log(factor))))
#sys.exit(0)

# 1b. compute and write allowed lengths
#start_dur = 0.88
#end_dur = 19.00
durs = []
d = start_dur
f = open(os.path.join(args.dir, 'allowed_durs.txt'), 'wb')
f2 = open(os.path.join(args.dir, 'allowed_lengths.txt'), 'wb')
while d < end_dur:
  length = int(d*1000 - 25) / 10 + 1  # for the most common length of frames and overlap
  if length % 3 != 0:
    lo = 3 * (length / 3)
    hi = lo + 3
    #if length - lo <= hi - length:
    #  length = lo
    #else:
    #  length = hi
    length = lo  # should select lo to make sure the jump is not bigger than 12%
    dnew = (10.0 * (length - 1.0) + 25.0 + 5.0) / 1000.0  # +5 is for safety
    d = dnew
  durs += [d]
  f.write(str(d) + '\n')
  f2.write(str(length) + '\n')
  d *= factor
f.close()
f2.close()

# 2. perturb to allowed durs
# sox -t wav seg1.wav -t wav long95.wav speed 0.873684211
perturbed_utts = []
durs = durs + [1000000]
for u in utts:
  prev_d = 0.0
  i = 0
  for d in durs:
    if u.dur <= d and u.dur >= prev_d:
      break
    i += 1
    prev_d = d
  # i determines the closest allowed durs

  if i > 0:
    allowed_dur = durs[i - 1]   # this is smaller than u.dur
    speed = u.dur / allowed_dur
    if max(speed, 1.0/speed) > factor:
      #print('rejected: {}    --> dur was {} speed was {}'.format(u.id, u.dur, speed))
      continue
    u1 = copy.deepcopy(u)
    u1.id = 'pv1-' + u.id
    u1.speaker = 'pv1-' + u.speaker
    u1.wavefile = '{} sox -t wav - -t wav - speed {} | '.format(u.wavefile, speed)
    u1.dur = allowed_dur
    if not args.no_speed_perturb:
      perturbed_utts += [u1]


  if i < len(durs) - 1:
    allowed_dur2 = durs[i]  # this is bigger than u.dur
    speed = u.dur / allowed_dur2
    if max(speed, 1.0/speed) > factor:
      #print('no v2/v3 for: {}    --> dur was {} speed was {}'.format(u.id, u.dur, speed))
      continue

    ## Add two versions for the second allowed_length
    ## one version is by using speed modification using sox
    ## the other is by extending by silence
    u2 = copy.deepcopy(u)
    u2.id = 'pv2-' + u.id
    u2.speaker = 'pv2-' + u.speaker
    u2.wavefile = '{} sox -t wav - -t wav - speed {} | '.format(u.wavefile, speed)
    u2.dur = allowed_dur2
    if not args.no_speed_perturb:
      perturbed_utts += [u2]

    delta = allowed_dur2 - u.dur
    if delta <= 1e-4:
      continue
    u3 = copy.deepcopy(u)
    u3.id = 'pv3-' + u.id
    u3.speaker = 'pv3-' + u.speaker
    u3.wavefile = '{} extend-wav-with-silence --extra-silence-length={} - - | '.format(u.wavefile, delta)
    u3.dur = allowed_dur2
    perturbed_utts += [u3]

# 3. write to our dir
generate_kaldi_data_files(perturbed_utts, args.dir)
