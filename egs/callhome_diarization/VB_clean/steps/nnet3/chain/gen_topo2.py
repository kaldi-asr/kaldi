#!/usr/bin/env python

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)

# Generate a topology file.  This allows control of the number of states in the
# non-silence HMMs, and in the silence HMMs.  This is a modified version of
# 'utils/gen_topo.pl' that generates a different type of topology, one that we
# believe should be useful in the 'chain' model.  Note: right now it doesn't
# have any real options, and it treats silence and nonsilence the same.  The
# intention is that you write different versions of this script, or add options,
# if you experiment with it.

from __future__ import print_function
import argparse


parser = argparse.ArgumentParser(description="Usage: steps/nnet3/chain/gen_topo.py "
                                             "<colon-separated-nonsilence-phones> <colon-separated-silence-phones>"
                                             "e.g.:  steps/nnet3/chain/gen_topo.pl 4:5:6:7:8:9:10 1:2:3\n",
                                 epilog="See egs/swbd/s5c/local/chain/train_tdnn_a.sh for example of usage.");
parser.add_argument("nonsilence_phones", type=str,
                    help="List of non-silence phones as integers, separated by colons, e.g. 4:5:6:7:8:9");
parser.add_argument("silence_phones", type=str,
                    help="List of silence phones as integers, separated by colons, e.g. 1:2:3");

args = parser.parse_args()

silence_phones = [ int(x) for x in args.silence_phones.split(":") ]
nonsilence_phones = [ int(x) for x in args.nonsilence_phones.split(":") ]
all_phones = silence_phones +  nonsilence_phones

print("<Topology>")
print("<TopologyEntry>")
print("<ForPhones>")
print(" ".join([str(x) for x in all_phones]))
print("</ForPhones>")

# the pdf-classes are as follows:
#  pdf-class 0 is in a 1-frame sequence, the initial and final state.
#  pdf-class 1 is in a sequence with >=3 frames, the 'middle' states.  (important that
#   it be numbered 1, which is the default list of pdf-classes used in 'cluster-phones').
#  pdf-class 2 is the initial-state in a sequence with >= 2 frames.
#  pdf-class 3 is the final-state in a sequence with >= 2 frames.
# state 0 is nonemitting in this topology.

print("<State> 0 <Transition> 1 0.5 <Transition> 2 0.5 </State>")  # initial nonemitting state.
print("<State> 1 <PdfClass> 0 <Transition> 5 1.0 </State>")  # 1-frame sequence.
print("<State> 2 <PdfClass> 2 <Transition> 3 0.5 <Transition> 4 0.5 </State>")  # 2 or more frames
print("<State> 3 <PdfClass> 1 <Transition> 3 0.5 <Transition> 4 0.5 </State>")  # 3 or more frames
print("<State> 4 <PdfClass> 3 <Transition> 5 1.0 </State>") # 2 or more frames.
print("<State> 5 </State>")  # final nonemitting state

print("</TopologyEntry>")
print("</Topology>")

