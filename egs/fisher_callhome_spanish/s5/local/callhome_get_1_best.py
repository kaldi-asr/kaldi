#!/usr/bin/env python

# Copyright 2014  Gaurav Kumar.   Apache 2.0
# Extracts one best output for a set of files
# The list of files in the conversations for which 1 best output has to be extracted
# words.txt

import os
import sys

def findTranscription(timeDetail):
  file1 = open('exp/tri5a/decode_callhome_dev/scoring/13.tra')
  file2 = open('exp/tri5a/decode_callhome_train/scoring/13.tra')
  for line in file1:
    lineComp = line.split()
    if lineComp[0] == timeDetail:
      return " ".join(lineComp[1:])
  for line in file2:
    lineComp = line.split()
    if lineComp[0] == timeDetail:
      return " ".join(lineComp[1:])
  # No result found
  return -1


wordsFile = open('exp/tri5a/graph/words.txt')
words = {}

# Extract word list
for line in wordsFile:
  lineComp = line.split()
  words[int(lineComp[1])] = lineComp[0].strip()

# Now read list of files in conversations
fileList = []
#conversationList = open('/export/a04/gkumar/corpora/fishcall/joshkal-splits/provisional_dev')
conversationList = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-callhome/train')
for line in conversationList:
    line = line.strip()
    line = line[:-4]
    fileList.append(line)

# IN what order were the conversations added to the spanish files?
# TODO: Make sure they match the order in which these english files are being written

# Now get timing information to concatenate the ASR outputs
if not os.path.exists('exp/tri5a/one-best/ch_train'):
  os.makedirs('exp/tri5a/one-best/ch_train')

#provFile = open('/export/a04/gkumar/corpora/fishcall/fisher_provisional_dev.es', 'w+')
provFile = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-callhome/asr.train', 'w+')
for item in fileList:
  timingFile = open('/export/a04/gkumar/corpora/fishcall/callhome/tim/' + item + '.es')
  newFile = open('exp/tri5a/one-best/ch_train/' + item + '.es', 'w+')
  for line in timingFile:
    timeInfo = line.split()
    mergedTranslation = ""
    for timeDetail in timeInfo:
      #Locate this in ASR dev/test, this is going to be very slow
      tmp = findTranscription(timeDetail)
      if tmp != -1:
        mergedTranslation = mergedTranslation + " " + tmp
    mergedTranslation = mergedTranslation.strip()
    transWords = [words[int(x)] for x in mergedTranslation.split()]
    newFile.write(" ".join(transWords) + "\n")
    provFile.write(" ".join(transWords) + "\n")

  newFile.close()
provFile.close()






