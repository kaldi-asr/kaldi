#!/usr/bin/env python

# Copyright 2017 (author: Chun-Chieh Chang)

# Generate a topology file.  This allows control of the number of states in the
# non-silence HMMs, and in the silence HMMs. This is a modified version of
# 'utils/gen_topo.pl'. The difference is that this creates two topologies for
# the non-silence HMMs. The number of states for punctuations is different than
# the number of states for other characters.

from __future__ import print_function
from __future__ import division
import argparse
import string

parser = argparse.ArgumentParser(description="Usage: steps/nnet3/chain/gen_topo.py "
                                             "<colon-separated-nonsilence-phones> <colon-separated-silence-phones>"
                                             "e.g.:  steps/nnet3/chain/gen_topo.pl 4:5:6:7:8:9:10 1:2:3\n",
                                 epilog="See egs/swbd/s5c/local/chain/train_tdnn_a.sh for example of usage.");
parser.add_argument("num_nonsil_states", type=int, help="number of states for nonsilence phones");
parser.add_argument("num_sil_states", type=int, help="number of states for silence phones");
parser.add_argument("num_punctuation_states", type=int, help="number of states for punctuation");
parser.add_argument("nonsilence_phones",
                    help="List of non-silence phones as integers, separated by colons, e.g. 4:5:6:7:8:9");
parser.add_argument("silence_phones",
                    help="List of silence phones as integers, separated by colons, e.g. 1:2:3");
parser.add_argument("phone_list", help="file containing all phones and their corresponding number.");

args = parser.parse_args()

silence_phones = [ int(x) for x in args.silence_phones.split(":") ]
nonsilence_phones = [ int(x) for x in args.nonsilence_phones.split(":") ]
all_phones = silence_phones +  nonsilence_phones

punctuation_phones = []
exclude = set("!(),.?;:'-\"")
with open(args.phone_list) as f:
    for line in f:
        line = line.strip()
        phone = line.split(' ')[0]
        if len(phone) == 1 and phone in exclude:
            punctuation_phones.append(int(line.split(' ')[1]))
# For nonsilence phones that are not punctuations
print("<Topology>")
print("<TopologyEntry>")
print("<ForPhones>")
print(" ".join([str(x) for x in nonsilence_phones if x not in punctuation_phones]))
print("</ForPhones>")
for x in range(0, args.num_nonsil_states):
    xp1 = x + 1
    print("<State> {0} <PdfClass> {0} <Transition> {0} 0.75 <Transition> {1} 0.25 </State>".format(x, xp1))
print("<State> {} </State>".format(args.num_nonsil_states))
print("</TopologyEntry>")

# For nonsilence phones that ar punctuations
print("<TopologyEntry>")
print("<ForPhones>")
print(" ".join([str(x) for x in nonsilence_phones if x in punctuation_phones]))
print("</ForPhones>")
for x in range(0, args.num_punctuation_states):
    xp1 = x + 1
    print("<State> {0} <PdfClass> {0} <Transition> {0} 0.75 <Transition> {1} 0.25 </State>".format(x, xp1))
print("<State> {} </State>".format(args.num_punctuation_states))
print("</TopologyEntry>")

# For silence phones
print("<TopologyEntry>")
print("<ForPhones>")
print(" ".join([str(x) for x in silence_phones]))
print("</ForPhones>")
if(args.num_sil_states > 1):
    transp = 1.0 / (args.num_sil_states - 1)
    
    state_str = "<State> 0 <PdfClass> 0 "
    for x in range(0, (args.num_sil_states - 1)):
        state_str = "{}<Transition> {} {} ".format(state_str, x, transp))
    state_str = state_str + "</State>"
    print(state_str)

    for x in range(1, (args.num_sil_states - 1)):
        state_str = "<State> {0} <PdfClass {0} ".format(x))
        for y in range(1, args.num_sil_states):
            state_str = "{}<Transition> {} {} ".format(state_str, y, transp))
        state_str = state_str + "</State>"
        print(state_str)
    second_last = args.num_sil_states - 1
    print("<State> {0} <PdfClass> {0} <Transition> {0} 0.75 <Transition> {1} 0.25 </State>".format(second_last, args.num_sil_states))
    print("<State> {} </State>".format(args.num_sil_states))
else:
    print("<State> 0 <PdfClass> 0 <Transition> 0 0.75 <Transition> 1 0.25 </State>")
    print("<State> {} </State>".format(args.num_sil_states))
print("</TopologyEntry>")
print("</Topology>")
