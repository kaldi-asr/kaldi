#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, io

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

# This script adds ligatures as pronunciation variants. We allow only one ligature
# per pronunciation but apply all possible rules

classMap = {
  'hh': 'x',
  'am': 'a',
  'ae': 'a',
  'ah': 'a',
  'al': 'a',
  'aa': 'a',
  'ba': 'b',
  'te': 'x',
  'ta': 'b',
  'th': 'b',
  'ja': 'h',
  'ha': 'h',
  'kh': 'h',
  'da': 'd',
  'dh': 'd',
  'ra': 'd',
  'zy': 'd',
  'se': 's',
  'sh': 's',
  'sa': 'o',
  'de': 'o',
  'to': 't',
  'za': 't',
  'ay': 'i',
  'gh': 'i',
  'fa': 'f',
  'ka': 'f',
  'ke': 'k',
  'la': 'l',
  'ma': 'm',
  'na': 'n',
  'he': 'x',
  'wa': 'x',
  'ee': 'j',
  'ya': 'j'
}

def match(phoneme, placeholder):
    if phoneme == placeholder:
        return True
    if len(phoneme) < 2 or len(placeholder) < 2:
        return False
    p = phoneme[:-1]
    if not p in classMap:
        return False
    return (phoneme[-1:] == placeholder[-1:]) and (classMap[p] == placeholder[:-1])

# Load ligature file
rules = dict()
with open(sys.argv[1], encoding="utf-8") as f:
    for x in f:
        parts = x.strip().split()
        if len(parts) < 2 or parts[0].startswith('#'):
            continue
        name = parts.pop(0)
        if name not in rules:
            rules[name] = []
        rules[name].append(parts)

# Read stdin
in_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
out_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
for line in in_stream:
    out_stream.write(line)
    phonemes = line.strip().split()
    word = phonemes.pop(0)
    for start in range(0, len(phonemes) - 1):
        if phonemes[start] == 'conn' or phonemes[start] == 'sil':
            continue
        for ruleName in rules:
            for variant in rules[ruleName]:
                matched = True
                for offset in range(0, len(variant)):
                    if not match(phonemes[start+2*offset], variant[offset]):
                        matched = False
                        break
                if matched:
                    out_stream.write(word + " " 
                            + ((' '.join(phonemes[0:start])) + ' '
                            + ruleName + ' '
                            + (' '.join(phonemes[start+2*offset+1:]))).strip() + "\n")
                    break
