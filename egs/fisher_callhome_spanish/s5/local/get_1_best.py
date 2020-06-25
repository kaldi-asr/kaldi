#!/usr/bin/env python
# Copyright 2014  Gaurav Kumar.   Apache 2.0

# Extracts one best output for a set of files
# The list of files in the conversations for which 1 best output has to be extracted
# words.txt

import os
import sys

scoringFile = "exp/sgmm2x_6a_mmi_b0.2/decode_test_it4/scoring/10.tra"
wordsFile = open('exp/sgmm2x_6a/graph/words.txt')
conversationList = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-matt/test')
oneBestTmp = 'exp/sgmm2x_6a_mmi_b0.2/one-best/asr-test'
provFile = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-matt/asr.test', 'w+')
timLocation = '/export/a04/gkumar/corpora/fishcall/fisher/tim'

def findTranscription(timeDetail):
  file1 = open(scoringFile)
  for line in file1:
    lineComp = line.split()
    if lineComp[0] == timeDetail:
      return " ".join(lineComp[1:])
  # No result found
  return -1

words = {}

# Extract word list
for line in wordsFile:
  lineComp = line.split()
  words[int(lineComp[1])] = lineComp[0].strip()

# Now read list of files in conversations
fileList = []
for line in conversationList:
    line = line.strip()
    line = line[:-4]
    fileList.append(line)

# Now get timing information to concatenate the ASR outputs
if not os.path.exists(oneBestTmp):
  os.makedirs(oneBestTmp)

for item in fileList:
  timingFile = open(timLocation + '/' + item + '.es')
  newFile = open(oneBestTmp + '/' + item + '.es', 'w+')
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
