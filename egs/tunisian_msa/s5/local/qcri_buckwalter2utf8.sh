#!/bin/bash

cut -d " " -f 1 qcri.txt > qcri_words_buckwalter.txt
cut -d " " -f 2 qcri.txt > qcri_prons.txt

local/buckwalter2unicode.py -i qcri_words_buckwalter.txt -o qcri_words_utf8.txt

paste qcri_words_utf8.txt qcri_prons.txt

rm qcri_words_buckwalter.txt qcri_words_utf8.txt qcri_prons.txt
