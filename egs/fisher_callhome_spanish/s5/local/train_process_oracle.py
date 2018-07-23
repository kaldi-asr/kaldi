#!/usr/bin/env python
# Copyright 2014  Gaurav Kumar.   Apache 2.0

import os
import sys

files = [
open('/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-1/exp/tri5a/decode_test/oracle/oracle.tra'),
open('/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-2/exp/tri5a/decode_test/oracle/oracle.tra'),
open('/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-3/exp/tri5a/decode_test/oracle/oracle.tra'),
open('/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-4/exp/tri5a/decode_test/oracle/oracle.tra'),
open('/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-5/exp/tri5a/decode_test/oracle/oracle.tra'),
open('/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-6/exp/tri5a/decode_test/oracle/oracle.tra'),
open('/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-7/exp/tri5a/decode_test/oracle/oracle.tra'),
open('/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-8/exp/tri5a/decode_test/oracle/oracle.tra'),
open('/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-9/exp/tri5a/decode_test/oracle/oracle.tra'),
open('/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-10/exp/tri5a/decode_test/oracle/oracle.tra')]

def findTranscription(timeDetail):

    for file1 in files:
        file1.seek(0,0)
        for line in file1:
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
conversationList = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-matt/train')
for line in conversationList:
    line = line.strip()
    line = line[:-4]
    fileList.append(line)

# IN what order were the conversations added to the spanish files?
# TODO: Make sure they match the order in which these english files are being written

# Now get timing information to concatenate the ASR outputs
if not os.path.exists('exp/tri5a/one-best/train'):
    os.makedirs('exp/tri5a/one-best/train')

#provFile = open('/export/a04/gkumar/corpora/fishcall/fisher_provisional_dev.es', 'w+')
provFile = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-matt/asr.train.oracle', 'w+')
for item in fileList:
    timingFile = open('/export/a04/gkumar/corpora/fishcall/fisher/tim/' + item + '.es')
    newFile = open('exp/tri5a/one-best/train/' + item + '.es', 'w+')
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






