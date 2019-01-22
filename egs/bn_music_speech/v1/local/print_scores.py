#!/usr/bin/env python
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script prints out lines of the form:
# <score> <target-type>.
# Its output is meant to be used as input to the binary
# compute-eer. The Broadcast News utterances have either
# "music" or "speech" in the utterance name, and so we
# can simply check if the utterance name contains  one of
# those strings to determine if it is a target or nontarget
# utterance. We arbitrarily pick music to be the target class.

from __future__ import print_function
import sys
utt2score = open(sys.argv[1], 'r').readlines()
for i in range(0, len(utt2score)):
  utt, score = utt2score[i].rstrip().split()
  if "music" in utt:
    type = "target"
  else:
    type = "nontarget"
  print(score, type)
