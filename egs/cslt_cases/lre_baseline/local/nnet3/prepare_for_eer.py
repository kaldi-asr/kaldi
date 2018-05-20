#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.


import sys


trials = open(sys.argv[1], 'r').readlines()
scores = open(sys.argv[2], 'r').readlines()

langutt2target = {}
for line in trials:
  lang, utt, target = line.strip().split()
  langutt2target[lang+utt]=target

langs_order = {} # langs order in the first line of scores
langs = scores[0].strip().split()
for i in range(len(langs)):
  langs_order[i] = langs[i]

for line in scores[1:]:
  items = line.strip().split()
  utt = items[0]
  sco = items[1:]
  for i in range(len(sco)):
    if langutt2target.has_key(langs_order[i]+utt):
      print sco[i], langutt2target[langs_order[i]+utt]

