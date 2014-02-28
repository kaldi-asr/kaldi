#!/usr/bin/env python
# Copyright 2014 David Snyder.  
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# This script creates a table between language names
# and integers. The first argument is an utt2lang
# file and the second the write location of the table.

import sys
utt2lang = open(sys.argv[1], 'r')
lang_map = {}
count = 0
for line in utt2lang.readlines():
  utt, lang = line.split(" ")
  lang = lang.strip()
  if lang not in lang_map:
    count += 1
    lang_map[lang] = count
table = ""
for lang in lang_map:
  table = table + lang + " " + str(lang_map[lang]) + "\n"
lang_table = open(sys.argv[2], 'w')
lang_table.write(table)
  
  
