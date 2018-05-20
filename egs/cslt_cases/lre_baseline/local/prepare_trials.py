#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)


import sys, collections

if len(sys.argv) != 3:
  print 'usage: prepare_trials.py <enroll-dir> <test-dir>'
  sys.exit()

enroll_dir = sys.argv[1]
test_dir = sys.argv[2]

# lang list in enroll dir
lang_list = []
with open(enroll_dir + '/utt2lang', 'r') as langs:
  for line in langs:
    lang_list.append(line.strip().split()[1])
lang_list = list(set(lang_list))
lang_list.sort()

# utt2lang dict in test dir
lang_dict = collections.OrderedDict()
with open(test_dir + '/utt2lang', 'r') as lang_ids:
  for lang_id in [line.strip().split() for line in lang_ids]:
     lang_dict[lang_id[0]] = lang_id[1]

# generate trials
trial = open(test_dir + '/trials', 'w')
for i in lang_dict.keys():
  for j in lang_list:
    if j == lang_dict[i]:
      trial.write(j + ' ' + i + ' ' + 'target' + '\n')
    else:
      trial.write(j + ' ' + i + ' ' + 'nontarget' + '\n')

print('Finished preparing trials.')
