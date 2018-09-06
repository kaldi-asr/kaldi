#!/usr/bin/env python3
# Copyright      2018  Ashish Arora
# Apache 2.0
# This script reads valid phones and removes the lines in the corpus
# which have any other phone.

import os
import sys, io

phone_file = os.path.join('data/local/text/cleaned/phones.txt')
infile = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
phone_dict = dict()
with open(phone_file, 'r', encoding='utf-8') as phone_fh:
    for line in phone_fh:
        line = line.strip().split()[0]
        phone_dict[line] = line

phone_dict[' '] = ' '
corpus_text = list()
for line in infile:
    text = line.strip()
    skip_text = False
    for phone in text:
        if phone not in phone_dict.keys():
            skip_text = True
            break
    if not skip_text:
        output.write(text+ '\n')

