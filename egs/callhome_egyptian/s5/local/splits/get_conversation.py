#!/usr/bin/env python

from __future__ import print_function
import os
import re

transdir = '/export/a04/gkumar/corpora/ECA/callhome/LDC97T19/callhome_arabic_trans_970711/transcrp'
devtest = {}
evaltest = {}
train = {}
devConv = 0
testConv = 0
trainConv = 0

pattern = re.compile(r"\d+\.\d*\s\d+\.\d*\s[AB]")

for root, _, files in os.walk(transdir):
  for f in files:
    if ".scr" in f:
      fullpath = os.path.join(root, f)
      pathComponents = fullpath.split("/")

      # Get all conversations
      trans = open(fullpath)
      numberOfConversations = 0
      for line in trans:
        if re.match(pattern, line):
          numberOfConversations = numberOfConversations + 1
      trans.close()

      if pathComponents[10] == 'devtest':
        devtest[pathComponents[12]] = numberOfConversations
        devConv = devConv + numberOfConversations
      if pathComponents[10] == 'train':
        train[pathComponents[12]] = numberOfConversations
        trainConv = trainConv + numberOfConversations
      if pathComponents[10] == 'evaltest':
        evaltest[pathComponents[12]] = numberOfConversations
        testConv = testConv + numberOfConversations

print("==============Train===============")
print(train)
print("Total Conversations in train = {}".format(trainConv))
print("==============Dev===============")
print(devtest)
print("Total Conversations in dev = {}".format(devConv))
print("==============Test===============")
print(evaltest)
print("Total Conversations in test = {}".format(testConv))
print("=================================")
print("Total Conversations in Corpus = {}".format(trainConv + devConv + testConv))
