#!/usr/bin/env python3

# Copyright  2020   Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0.

"""This script modifies TS-VAD output probabilities applying 
absolute threshold (--threshold) and relative threshold (--multispk_threshold) for pi/(p1+p2+p3+p4)  
(to exclude overlapping regions from i-vectors estimation)"""

import os
import argparse
import regex as re
import numpy as np
from scipy import signal, ndimage
from kaldiio import ReadHelper, WriteHelper

class WeightsSet:
    def __init__(self, vad_rspec, reg_exp):
        data = dict()
        prev = -1
        with ReadHelper(vad_rspec) as reader:
            for utid, align in reader:
                result = reg_exp.match(utid)
                assert result is not None, 'Wrong VAD alignment utterance ID format: \"{}\"'.format(utid)
                sess = result.group(1)
                piece = result.group(2)
                spkr = result.group(3)
                if sess not in data.keys():
                    data[sess] = dict()
                if piece not in data[sess].keys():
                    data[sess][piece] = dict()
                data[sess][piece][spkr]=align
            reader.close()
        print('  loaded {} sessions'.format(len(data)))
        self.data = data

    def modify_prob(self, threshold, multispk_threshold, lowest_value):
        for sess in self.data.keys():
            for piece in self.data[sess].keys():
                maxlen=0
                longest=""
                for spkr in self.data[sess][piece].keys():
                    if (len(self.data[sess][piece][spkr]) > maxlen):
                        maxlen=len(self.data[sess][piece][spkr])
                        longest=spkr
                sumprob=self.data[sess][piece][longest].copy()
                for spkr in self.data[sess][piece].keys():
                    if spkr == longest:
                        continue
                    for i in range(len(self.data[sess][piece][spkr])):
                        sumprob[i]+=self.data[sess][piece][spkr][i]
                for spkr in self.data[sess][piece].keys():
                    for i in range(len(self.data[sess][piece][spkr])):
                        if (self.data[sess][piece][spkr][i] < threshold):
                            self.data[sess][piece][spkr][i]=lowest_value
                for spkr in self.data[sess][piece].keys():
                    for i in range(len(self.data[sess][piece][spkr])):
                        if (self.data[sess][piece][spkr][i]/sumprob[i] < multispk_threshold):
                            self.data[sess][piece][spkr][i]=lowest_value


    def write(self, vad_wspec):
        with WriteHelper(vad_wspec) as writer:
            for sess in self.data.keys():
                for piece in self.data[sess].keys():
                    for spkr in self.data[sess][piece].keys():
                        utt=sess+'-'+piece+'-'+spkr
                        writer(utt, self.data[sess][piece][spkr])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: vad_prob_mod.py <vad-rspec> <vad-wspec>')
    parser.add_argument("--reg_exp", "-x", type=str, default=r'^(S\d\d.*)\-(\d+)\-(\d)$')
    parser.add_argument("--threshold", "-t", type=float, default=0.0)
    parser.add_argument("--multispk_threshold", "-mt", type=float, default=0.8)
    parser.add_argument("--lowest_value", "-l", type=float, default=0.00001)
    parser.add_argument('vad_rspec', type=str)
    parser.add_argument('vad_wspec', type=str)
    args = parser.parse_args()

    print('Options:')
    print('  Utterance ID regexp: {}'.format(args.reg_exp))
    print('  Absolute threshold:     {}'.format(args.threshold))
    print('  Multispeaker threshold for Pi/(P1+P2+P3+P4):     {}'.format(args.multispk_threshold))
    print('  Lowest value which is used when applying the thresholds:    {}'.format(args.lowest_value))
    print('  VAD rspec:    {}'.format(args.vad_rspec))
    print('  VAD wspec:    {}'.format(args.vad_wspec))

    reg_exp = re.compile(args.reg_exp)

    print('Loading VAD probabilities')
    vad_align = WeightsSet(args.vad_rspec, reg_exp)

    print('Modifying VAD probabilities')
    vad_align.modify_prob(args.threshold, args.multispk_threshold, args.lowest_value)

    print('Writing VAD probabilities')
    vad_align.write(args.vad_wspec)
