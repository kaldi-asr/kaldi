#!/usr/bin/env python

import sys
import codecs

romanToScript = {}

lexiconLocation = sys.argv[1]
wordLexiconLoc = sys.argv[2]
outLexiconLoc = sys.argv[3]

# First create a map from the romanized to the script version
lexicon = codecs.open(lexiconLocation + "callhome_arabic_lexicon_991012/ar_lex.v07", \
    encoding="iso-8859-6")
for line in lexicon:
  lineComp = line.strip().split('\t')
  romanToScript[lineComp[0]] = lineComp[1]
lexicon.close()

# Now read the word lexicon and write out a script lexicon
wordLexicon = open(wordLexiconLoc)
outLexicon = codecs.open(outLexiconLoc, "w+", encoding="utf-8")
for line in wordLexicon:
  lineComp = line.strip().split(" ")
  if lineComp[0] in romanToScript:
    lineComp[0] = romanToScript[lineComp[0]]
  outLexicon.write(" ".join(lineComp) + '\n')

wordLexicon.close()
outLexicon.close()
