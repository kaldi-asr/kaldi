#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)


import sys, collections

if len(sys.argv) != 3:
  print 'usage: prepare_trials.py <enroll-dir> <test-dir>'
  sys.exit()

enroll_dir = sys.argv[1]
test_dir = sys.argv[2]

# spk list in enroll dir
spk_list = []
with open(enroll_dir + '/spk2utt', 'r') as spks:
  for line in spks:
    spk_list.append(line.strip().split()[0])

# utt2spk dict in test dir
spk_dict = collections.OrderedDict()
with open(test_dir + '/utt2spk', 'r') as spk_ids:
  for spk_id in [line.strip().split() for line in spk_ids]:
     spk_dict[spk_id[0]] = spk_id[1]

# generate trials
trial = open(test_dir + '/trials', 'w')
for i in spk_list:
  for j in spk_dict.keys():
    if i == spk_dict[j]:
      trial.write(i + ' ' + j + ' ' + 'target' + '\n')
    else:
      trial.write(i + ' ' + j + ' ' + 'nontarget' + '\n')

print('Finished preparing trials.')
