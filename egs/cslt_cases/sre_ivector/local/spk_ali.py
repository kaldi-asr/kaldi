#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)


import sys, subprocess, collections, os


def get_utt_len_dict(data):
  '''Get num of frames for each utt.
  '''
  cmd = '. ./path.sh; feat-to-len scp:' + data + '/feats.scp ark,t:|less'
  utts_lens = subprocess.check_output(cmd, shell=True)
  len_dict = collections.OrderedDict()
  for utt_len in utts_lens.strip().split('\n'):
    len_dict[utt_len.strip().split()[0]] = utt_len.strip().split()[1]
  return len_dict


def get_utt_len_dict_vad(data):
  '''Get num of frames for each utt with vad.
  '''
  cmd = '. ./path.sh; copy-vector scp:' + data + '/vad.scp ark,t:|less'
  vads = subprocess.check_output(cmd, shell=True)
  len_dict = collections.OrderedDict()
  for line in vads.strip().split('\n'):
    utt = line.strip().split()[0]
    vad = line.strip().split()[1:]
    num = 0
    for i in vad[1:len(vad)-1]:
      num += int(i)
    len_dict[utt] = num
  return len_dict


def get_spk_id_dict(data, dir):
  '''Get spk id for each utt and spk num. Store spk2spk_id in dir.
  '''
  id = 0 # int id starts from 0
  spk_dict = {}
  min = 10000 # spoken by one spk
  max = 0 # spoken by one spk
  content = '' # spk2spk_id for writing
  with open(data + '/spk2utt', 'r') as lines:
    for spk2utts in [line.strip().split() for line in lines]:
      if max < len(spk2utts) - 1:
        max = len(spk2utts) - 1
      if min > len(spk2utts) - 1:
        min = len(spk2utts) - 1
      for i in spk2utts[1:]:
        spk_dict[i] = id
      content += spk2utts[0] + ' ' + str(id) + '\n'
      id += 1     
  # store spk2spk_id in dir
  spk2spk_id = open(dir + '/spk2spk_id', 'w')
  spk2spk_id.write(content)
  spk2spk_id.close()
  print('max utts spoken by one person:')
  print(max)
  print('min utts spoken by one person:')
  print(min)
  return spk_dict, id


def generate_spk_ali(spk_dict, spk_num, len_dict, dir):
  '''Generate spk ali and store it in dir.
  '''
  content = '' # spk ali of each utt for writing
  counts = []  # total num of frames per spk
  for n in range(0, spk_num):
    counts.append(0)
  for i in len_dict.keys():
    content += i + (' ' + str(spk_dict[i])) * int(len_dict[i]) + '\n'
    counts[int(spk_dict[i])] += int(len_dict[i])
  # write spk aligment, spk num, and total num of frames per spk
  ali = open(dir + '/ali.ark.tmp', 'w')
  num = open(dir + '/target_num', 'w')
  cou = open(dir + '/target_counts', 'w')
  ali.write(content)
  num.write(str(spk_num))
  cou.write('[')
  for j in counts:
    cou.write(' ' + str(j))
  cou.write(' ]')
  ali.close()
  num.close()
  cou.close()
  # write ali.ark with ali.scp
  cmd = ('. ./path.sh; copy-int-vector ark:' +
         dir + '/ali.ark.tmp ark,t,scp:' +
         dir + '/ali.ark,' + dir + '/ali.scp;' +
         'rm ' + dir + '/ali.ark.tmp')
  subprocess.check_call(cmd, shell=True)
  

if __name__ == "__main__":
  '''Get int spk id alignments on frame level for each utt with or without vad.
  '''
  if len(sys.argv) != 4 or (sys.argv[1] != '-vad' and sys.argv[1] != '-novad'):
    print('usage: spk_ali.py [-vad|-novad] <data-dir> <spk-ali-dir>')
    sys.exit()
  
  vad = sys.argv[1]
  data = sys.argv[2]
  dir = sys.argv[3]
  if not os.path.exists(dir):
    os.makedirs(dir)

  spk_dict, spk_num = get_spk_id_dict(data, dir)
  if vad == '-vad':
    len_dict = get_utt_len_dict_vad(data)
  else:
    len_dict = get_utt_len_dict(data)
  generate_spk_ali(spk_dict, spk_num, len_dict, dir)
  print('Spk ali done.')

