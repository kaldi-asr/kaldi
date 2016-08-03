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
# The next two lines may look like a bug, but they are as intended.  State 0 has
# no self-loop, it happens exactly once.  And it can go either to state 1 (with
# a self-loop) or to state 2, so we can have zero or more instances of state 1
# following state 0.
# We make the transition-probs 0.5 so they normalize, to keep the code happy.
# In fact, we always set the transition probability scale to 0.0 in the 'chain'
# code, so they are never used.
print("<State> 0 <PdfClass> 0 <Transition> 1 0.5 <Transition> 2 0.5 </State>")
print("<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>")
print("<State> 2 </State>")
print("</TopologyEntry>")
print("</Topology>")

