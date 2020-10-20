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
with open(sys.argv[1], 'r') as u2s_fi:
  for line in u2s_fi:
    utt, score = line.rstrip().split()
    print(score, "target" if "music" in utt else "nontarget")
