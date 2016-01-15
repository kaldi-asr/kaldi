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
# state 0 is obligatory (occurs once)
print("<State> 0 <PdfClass> 0 <Transition> 1 0.3333 <Transition> 2 0.3333 <Transition> 3 0.3333 </State> ")
# state 1 is used only when >2 frames
print("<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>")
# state 2 is used only when >=2 frames (and occurs once)
print("<State> 2 <PdfClass> 2 <Transition> 3 1.0 </State>")
print("<State> 3 </State>")  # final nonemitting state
print("</TopologyEntry>")
print("</Topology>")

