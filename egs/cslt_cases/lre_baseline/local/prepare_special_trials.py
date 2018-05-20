#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)


import sys, collections

if len(sys.argv) != 4:
  print 'usage: prepare_special_trials.py <enroll-dir> <lang-list> <test-dir>'
  sys.exit()

enroll_dir = sys.argv[1]
lang_list = sys.argv[2].strip().split()
test_dir = sys.argv[3]

# utt2lang dict in test dir
lang_dict = collections.OrderedDict()
with open(test_dir + '/utt2lang', 'r') as lang_ids:
  for lang_id in [line.strip().split() for line in lang_ids]:
     if lang_id[1] in lang_list:
       lang_dict[lang_id[0]] = lang_id[1]

# generate trials
trial = open(test_dir + '/trials.' + '_'.join(lang_list), 'w')
for i in lang_dict.keys():
  for j in lang_list:
    if j == lang_dict[i]:
      trial.write(j + ' ' + i + ' ' + 'target' + '\n')
    else:
      trial.write(j + ' ' + i + ' ' + 'nontarget' + '\n')

#print('Finished preparing trials.')
