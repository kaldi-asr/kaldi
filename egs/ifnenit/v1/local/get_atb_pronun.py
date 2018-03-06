#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys

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

for line in sys.stdin:
    connected = False
    lastChar = ''
    lastType = ''
    sys.stdout.write(line.strip())
    for char in line.strip():
        if char == '+':
            continue
        if char == '=':
            connected = True
            continue
        sys.stdout.write((" " if lastChar else "") + lastChar + lastType + (" conn" if connected else " sil"))
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
    sys.stdout.write(" "+lastChar+lastType)
    sys.stdout.write("\n")
