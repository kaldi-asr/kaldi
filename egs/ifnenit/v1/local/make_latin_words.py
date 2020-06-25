#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

# Convert unicode words to position dependent latin form.
# This script make creating lexicon very easy. 

import os, sys, io

map = {
    'ء': 'hh',
    'آ': 'am',
    'أ': 'ae',
    'إ': 'ah',
    'ئ': 'al',
    'ا': 'aa',
    'ب': 'ba',
    'ة': 'te',
    'ت': 'ta',
    'ث': 'th',
    'ج': 'ja',
    'ح': 'ha',
    'خ': 'kh',
    'د': 'da',
    'ذ': 'dh',
    'ر': 'ra',
    'ز': 'zy',
    'س': 'se',
    'ش': 'sh',
    'ص': 'sa',
    'ض': 'de',
    'ط': 'to',
    'ظ': 'za',
    'ع': 'ay',
    'غ': 'gh',
    'ف': 'fa',
    'ق': 'ka',
    'ك': 'ke',
    'ل': 'la',
    'م': 'ma',
    'ن': 'na',
    'ه': 'he',
    'و': 'wa',
    'ى': 'ee',
    'ي': 'ya',
}

connecting = {
  'hh': False,
  'am': False,
  'ae': False,
  'ah': False,
  'al': False,
  'aa': False,
  'ba': True,
  'te': False,
  'ta': True,
  'th': True,
  'ja': True,
  'ha': True,
  'kh': True,
  'da': False,
  'dh': False,
  'ra': False,
  'zy': False,
  'se': True,
  'sh': True,
  'sa': True,
  'de': True,
  'to': True,
  'za': True,
  'ay': True,
  'gh': True,
  'fa': True,
  'ka': True,
  'ke': True,
  'la': True,
  'ma': True,
  'na': True,
  'he': True,
  'wa': False,
  'ee': False,
  'ya': True
}

in_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
out_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
for line in in_stream:
    connected = False
    lastChar = ''
    lastType = ''
    out_stream.write(line.strip())
    for char in line.strip():
        if char == '+':
            continue
        if char == '=':
            connected = True
            continue
        out_stream.write((" " if lastChar else "") + lastChar + lastType + (" conn" if connected else " sil"))
        if char in map:
            lastChar = map[char]
            if connected:
                if connecting[lastChar]:
                    lastType="M"
                else:
                    lastType="E"
            else:
                if connecting[lastChar]:
                    lastType="B"
                else:
                    lastType="A"
            connected=connecting[lastChar]
        else: # Not in map
            if char == '#':
                lastChar = 'hash'
            elif char == '_':
                lastChar = 'uscore'
            elif char == '<':
                lastChar = 'ltchar'
            elif char == '>':
                lastChar = 'gtchar'
            else:
                lastChar = char
            lastType = "A"
            connected=False
    if lastType == "M":
        lastType = "E"
    elif lastType == "B":
        lastType = "A"
    out_stream.write(" "+lastChar+lastType)
    out_stream.write("\n")
