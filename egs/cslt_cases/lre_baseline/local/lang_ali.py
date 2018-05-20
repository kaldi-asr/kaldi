#!/usr/bin/env python
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


def get_lang_id_dict(data, dir):
  '''Get lang id for each utt and lang num. Store lang2lang_id in dir.
  '''
  id = 0 # int id starts from 0
  lang_dict = {}
  min = 10000 # of one lang
  max = 0 # of one lang
  content = '' # lang2lang_id for writing

  # get lang2utt from utt2lang 
  cmd = '. ./path.sh; utils/utt2spk_to_spk2utt.pl ' + data + '/utt2lang'
  lang_utt = subprocess.check_output(cmd, shell=True)
  lang_utt = lang_utt.strip().split('\n')
  lang_utt.sort()

  for lang2utts in [line.strip().split() for line in lang_utt]:
    if max < len(lang2utts) - 1:
      max = len(lang2utts) - 1
    if min > len(lang2utts) - 1:
      min = len(lang2utts) - 1
    for i in lang2utts[1:]:
      lang_dict[i] = id
    content += lang2utts[0] + ' ' + str(id) + '\n'
    id += 1     
  # store lang2lang_id in dir
  lang2lang_id = open(dir + '/lang2lang_id', 'w')
  lang2lang_id.write(content)
  lang2lang_id.close()
  print('max utts of one language:')
  print(max)
  print('min utts of one language:')
  print(min)
  return lang_dict, id


def generate_lang_ali(lang_dict, lang_num, len_dict, dir):
  '''Generate lang ali and store it in dir.
  '''
  content = '' # lang ali of each utt for writing
  counts = []  # total num of frames per lang
  for n in range(0, lang_num):
    counts.append(0)
  for i in len_dict.keys():
    content += i + (' ' + str(lang_dict[i])) * int(len_dict[i]) + '\n'
    counts[int(lang_dict[i])] += int(len_dict[i])
  # write lang aligment, lang num, and total num of frames per lang
  ali = open(dir + '/ali.ark.tmp', 'w')
  num = open(dir + '/target_num', 'w')
  cou = open(dir + '/target_counts', 'w')
  ali.write(content)
  num.write(str(lang_num))
  cou.write('[')
  for j in counts:
    cou.write(' ' + str(j))
  cou.write(' ]')
  ali.close()
  num.close()
  cou.close()
  # write ali.ark with ali.scp
  cmd = ('. ./path.sh; copy-int-vector ark:' +
         dir + '/ali.ark.tmp ark,scp:' +
         dir + '/ali.ark,' + dir + '/ali.scp;' +
         'rm ' + dir + '/ali.ark.tmp')
  subprocess.check_call(cmd, shell=True)
  

if __name__ == "__main__":
  '''Get int lang id alignments on frame level for each utt with or without vad.
  '''
  if len(sys.argv) != 4 or (sys.argv[1] != '-vad' and sys.argv[1] != '-novad'):
    print('usage: lang_ali.py [-vad|-novad] <data-dir> <lang-ali-dir>')
    sys.exit()
  
  vad = sys.argv[1]
  data = sys.argv[2]
  dir = sys.argv[3]
  if not os.path.exists(dir):
    os.makedirs(dir)

  lang_dict, lang_num = get_lang_id_dict(data, dir)
  if vad == '-vad':
    len_dict = get_utt_len_dict_vad(data)
  else:
    len_dict = get_utt_len_dict(data)
  generate_lang_ali(lang_dict, lang_num, len_dict, dir)
  print('Language ali done.')

