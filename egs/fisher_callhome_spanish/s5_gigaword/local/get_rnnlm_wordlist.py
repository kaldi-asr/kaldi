#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    2018  Saikiran Valluri, GoVivace inc.

import os, sys

if len(sys.argv) < 4:
    print( "Usage: python get_rnnlm_wordlist.py <ASR lexicon words> <POCOLM wordslist> <RNNLM wordslist output>")
    sys.exit()

lexicon_words = open(sys.argv[1], 'r')
pocolm_words = open(sys.argv[2], 'r')
rnnlm_wordsout = open(sys.argv[3], 'w')

line_count=0
lexicon=[]

for line in lexicon_words:
    lexicon.append(line.split()[0])
    rnnlm_wordsout.write(line.split()[0] + " " + str(line_count)+'\n')
    line_count = line_count + 1

for line in pocolm_words:
    if not line.split()[0] in lexicon:
        rnnlm_wordsout.write(line.split()[0] + " " + str(line_count)+'\n')
        line_count = line_count + 1

lexicon_words.close()
pocolm_words.close()
rnnlm_wordsout.close()
        
