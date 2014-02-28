#!/usr/bin/env python
# Copyright 2014 David Snyder.  
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# The script uses a table of languages to ints and an utt2lang
# file and outputs a file similar to utt2lang, but where the 
# language names are replaced with the integer labels found in
# the table.

import sys
languages = open(sys.argv[1], 'r')
lang_map = {}
for line in languages.readlines():
  lang, id = line.split(" ")
  lang = lang.rstrip()
  id = id.rstrip()
  lang_map[lang] = id

utt2lang = open(sys.argv[2], 'r')
utt2langint_str = ""
for line in utt2lang.readlines():
  utt, lang = line.split(" ")
  utt = utt.rstrip()
  lang = lang.rstrip()
  utt2langint_str = utt2langint_str + utt + " " + lang_map[lang] + "\n"

utt2langint = open(sys.argv[3], 'w')
utt2langint.write(utt2langint_str)
  
  
