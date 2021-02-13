# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

import os
import re
import json
from imblearn.over_sampling import RandomOverSampler


def round_score(score, floor=0.1, min_val=0, max_val=2):
    score = max(min(max_val, score), min_val)
    return round(score / floor) * floor


def load_phone_symbol_table(filename):
    if not os.path.isfile(filename):
        return None, None
    int2sym = {}
    sym2int = {}
    with open(filename, 'r') as f:
        for line in f:
            sym, idx = line.strip('\n').split('\t')
            idx = int(idx)
            int2sym[idx] = sym
            sym2int[sym] = idx
    return sym2int, int2sym


def load_human_scores(filename, floor=0.1):
    with open(filename) as f:
        info = json.load(f)
    score_of = {}
    phone_of = {}
    for utt in info:
        phone_num = 0
        for word in info[utt]['words']:
            assert len(word['phones']) == len(word['phones-accuracy'])
            for i, phone in enumerate(word['phones']):
                key = f'{utt}.{phone_num}'
                phone_num += 1
                phone_of[key] = re.sub(r'[_\d].*', '', phone)
                score_of[key] = round_score(word['phones-accuracy'][i], floor)
    return score_of, phone_of


def balanced_sampling(x, y):
    sampler = RandomOverSampler()
    return sampler.fit_resample(x, y)
