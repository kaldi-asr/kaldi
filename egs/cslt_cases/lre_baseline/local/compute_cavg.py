#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.


import sys
from math import *


def get_langid_dict(trials):
  ''' Get lang2lang_id, utt2lang_id dicts and lang nums, lang_id starts from 0. 
      Also return trial list.
  '''
  langs = []
  lines = open(trials, 'r').readlines()
  for line in lines:
    lang, utt, target = line.strip().split()
    langs.append(lang)

  langs = list(set(langs))
  langs.sort()
  lang2lang_id = {}
  for i in range(len(langs)):
    lang2lang_id[langs[i]] = i

  utt2lang_id = {}
  trial_list = {}
  for line in lines:
    lang, utt, target = line.strip().split()
    if target == 'target':
      utt2lang_id[utt] = lang2lang_id[lang]
    trial_list[lang + utt] = target

  return lang2lang_id, utt2lang_id, len(langs), trial_list


def process_pair_scores(scores, lang2lang_id, utt2lang_id, lang_num, trial_list):
  ''' Replace both lang names and utt ids with their lang ids,
      for unknown utt, just with -1. Also return the min and max scores.
  '''
  pairs = []
  stats = []
  lines = open(scores, 'r').readlines()
  for line in lines:
    lang, utt, score = line.strip().split()
    if trial_list.has_key(lang + utt):
      if utt2lang_id.has_key(utt):
        pairs.append([lang2lang_id[lang], utt2lang_id[utt], float(score)])
      else:
        pairs.append([lang2lang_id[lang], -1, float(score)])
      stats.append(float(score))
  return pairs, min(stats), max(stats)


def process_matrix_scores(scores, lang2lang_id, utt2lang_id, lang_num, trial_list):
  ''' Convert matrix scores to pairs as returned by process_pair_scores.
  '''
  lines = open(scores, 'r').readlines()
  langs_order = {} # langs order in the first line of scores
  langs = lines[0].strip().split()
  for i in range(len(langs)):
    langs_order[i] = langs[i]

  pairs = []
  stats = []
  for line in lines[1:]:
    items = line.strip().split()
    utt = items[0]
    sco = items[1:]
    for i in range(len(sco)):
      if trial_list.has_key(langs_order[i] + utt):
        if utt2lang_id.has_key(utt):
          pairs.append([lang2lang_id[langs_order[i]], utt2lang_id[utt], float(sco[i])])
        else:
          pairs.append([lang2lang_id[langs_order[i]], -1, float(sco[i])])
        stats.append(float(sco[i]))
  return pairs, min(stats), max(stats)


def get_cavg(pairs, lang_num, min_score, max_score, bins = 20, p_target = 0.5):
  ''' Compute Cavg, using several threshhold bins in [min_score, max_score].
  '''
  cavgs = [0.0] * (bins + 1)
  precision = (max_score - min_score) / bins
  for section in range(bins + 1):
    threshold = min_score + section * precision
    # Cavg for each lang: p_target * p_miss + sum(p_nontarget*p_fa)
    target_cavg = [0.0] * lang_num
    for lang in range(lang_num):
      p_miss = 0.0 # prob of missing target pairs
      LTa = 0.0 # num of all target pairs
      LTm = 0.0 # num of missing pairs
      p_fa = [0.0] * lang_num # prob of false alarm, respect to all other langs
      LNa = [0.0] * lang_num # num of all nontarget pairs, respect to all other langs
      LNf = [0.0] * lang_num # num of false alarm pairs, respect to all other langs
      for line in pairs:
        if line[0] == lang:
          if line[1] == lang:
            LTa += 1
            if line[2] < threshold:
              LTm += 1
          else:
            LNa[line[1]] += 1
            if line[2] >= threshold:
              LNf[line[1]] += 1
      if LTa != 0.0:
        p_miss = LTm / LTa
      for i in range(lang_num):
        if LNa[i] != 0.0:
          p_fa[i] = LNf[i] / LNa[i]
      p_nontarget = (1 - p_target) / (lang_num - 1)
      target_cavg[lang] = p_target * p_miss + p_nontarget*sum(p_fa)
    cavgs[section] = sum(target_cavg) / lang_num

  return cavgs, min(cavgs)


if __name__ == '__main__':
  ''' Compute average cost for language recognition,
      see https://arxiv.org/pdf/1706.09742.pdf for details.

      Support two kinds of score formats, one is in pairs (each utt compared to all langs) as:
              lang_name_1  utt_id_1  0.01
              lang_name_2  utt_id_1  0.30
              ... ...
      the other one is in matrix (first row shows all language names), as:
                       lang_name_1 lang_name_2 ...
              utt_id_1    0.01        0.30     ...
              utt_id_2    0.31        0.20     ...
                 ...             ...
  '''
  if len(sys.argv) != 4 or (sys.argv[1] != '-pairs' and sys.argv[1] != '-matrix'):
    print('usage: compute_cavg.py [-pairs|-matrix] trials scores')
    sys.exit()

  form = sys.argv[1]
  trials = sys.argv[2]
  scores = sys.argv[3]

  lang2lang_id, utt2lang_id, lang_num, trial_list = get_langid_dict(trials)
  if form == '-pairs':
    pairs, min_score, max_score = process_pair_scores(scores, lang2lang_id, utt2lang_id, lang_num, trial_list) 
  else:
    pairs, min_score, max_score = process_matrix_scores(scores, lang2lang_id, utt2lang_id, lang_num, trial_list)

  threshhold_bins = 20
  p_target = 0.5
  cavgs, min_cavg = get_cavg(pairs, lang_num, min_score, max_score, threshhold_bins, p_target)

  print(round(min_cavg, 4))

