#!/usr/bin/env python3

import os
import sys, io

infile = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
phone_dict = dict()
for line in infile:
    line_vect = line.strip().split()
    for word in line_vect:
        for phone in word:
            phone_dict[phone] = phone

for phone in phone_dict.keys():
        output.write(phone+ '\n')
