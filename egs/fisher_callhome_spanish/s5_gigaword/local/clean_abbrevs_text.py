#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    2018  Saikiran Valluri, GoVivace inc.,

import os, sys
import re
import codecs

if len(sys.argv) < 3:
  print("Usage : python clean_abbrevs_text.py <Input text> <output text>")
  print("        Processes the text before text normalisation to convert uppercase words as space separated letters")
  sys.exit()

inputfile=codecs.open(sys.argv[1], encoding='utf-8')
outputfile=codecs.open(sys.argv[2], encoding='utf-8', mode='w')

for line in inputfile:
  words = line.split()
  textout = ""
  wordcnt = 0
  for word in words:
    if re.match(r"\b([A-ZÂÁÀÄÊÉÈËÏÍÎÖÓÔÖÚÙÛÑÇ])+[']?s?\b", word) and wordcnt>0:
      print(word)
      word = re.sub('\'?s', 's', word)
      textout = textout + " ".join(word) + " "
    else:
      textout = textout + word + " "
      if word.isalpha():
        wordcnt = wordcnt + 1
  outputfile.write(textout.strip()+ '\n')

inputfile.close()
outputfile.close() 
