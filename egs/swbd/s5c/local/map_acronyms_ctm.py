#!/usr/bin/env python

# Copyright 2015  Minhua Wu
# Apache 2.0

# convert acronyms in swbd decode result
# e.g. convert things like en_4156 B 414.26 0.65 u._c._l._a. to
# en_4156 B 414.26 0.16 u
# en_4156 B 414.42 0.16 c
# en_4156 B 414.58 0.16 l
# en_4156 B 414.74 0.17 a

from __future__ import division
import argparse,re
__author__ = 'Minhua Wu'
 
parser = argparse.ArgumentParser(description='convert acronyms back to eval2000 format')
parser.add_argument('-i','--input', help='Input ctm file ',required=True)
parser.add_argument('-o','--output',help='Output ctm file', required=True)
parser.add_argument('-M','--Map',help='Input acronyms map', required=True)
args = parser.parse_args()

if args.input == '-': args.input = '/dev/stdin'
if args.output == '-': args.output = '/dev/stdout'

dict_acronym_back = {}
fin_map = open(args.Map, "r")
for line in fin_map:
    items = line.split('\t')
    dict_acronym_back[items[1]] = items[2]

fin_map.close()

fin = open(args.input,"r")
fout = open(args.output, "w")

for line in fin:
    items = line.split()
    
    if items[4] in dict_acronym_back.keys():
        letters = dict_acronym_back[items[4]].split()
        acronym_period = round(float(items[3]), 2)
        letter_slot = round(acronym_period / len(letters), 2)
        time_start = round(float(items[2]), 2)
        for l in letters[:-1]:
            time = " %.2f %.2f " % (time_start, letter_slot)
            fout.write(' '.join(items[:2])+ time + l + "\n")
            time_start = time_start + letter_slot
        last_slot = acronym_period - letter_slot * (len(letters) - 1)
        time = " %.2f %.2f " % (time_start, last_slot)
        fout.write(' '.join(items[:2])+ time + letters[-1] + "\n")
    else:
        fout.write(line)    


