#!/usr/bin/env bash

# Copyright 2018 John Morgan
# Apache 2.0.

# write separate files for word and pronunciation fields
cut -d " " -f 1 qcri.txt > qcri_words_buckwalter.txt
cut -d " " -f 2- qcri.txt > qcri_prons.txt

# convert words to utf8 
local/buckwalter2unicode.py -i qcri_words_buckwalter.txt -o qcri_words_utf8.txt

paste qcri_words_utf8.txt qcri_prons.txt

rm qcri_words_buckwalter.txt qcri_words_utf8.txt qcri_prons.txt
