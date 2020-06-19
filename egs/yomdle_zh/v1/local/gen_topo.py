#!/usr/bin/env python

# Copyright 2017 (author: Chun-Chieh Chang)

# Generate a topology file.  This allows control of the number of states in the
# non-silence HMMs, and in the silence HMMs. This is a modified version of
# 'utils/gen_topo.pl'. The difference is that this creates two topologies for
# the non-silence HMMs. The number of states for punctuations is different than
# the number of states for other characters.

from __future__ import print_function
import argparse
import string

parser = argparse.ArgumentParser(description="Usage: steps/nnet3/chain/gen_topo.py "
                                             "<colon-separated-nonsilence-phones> <colon-separated-silence-phones>"
                                             "e.g.:  steps/nnet3/chain/gen_topo.pl 4:5:6:7:8:9:10 1:2:3\n",
                                 epilog="See egs/swbd/s5c/local/chain/train_tdnn_a.sh for example of usage.");
parser.add_argument("num_nonsil_states", type=int, help="number of states for nonsilence phones");
parser.add_argument("num_sil_states", type=int, help="number of states for silence phones");
parser.add_argument("num_cj5_states", type=int, help="number of states for punctuation");
parser.add_argument("nonsilence_phones", type=str,
                    help="List of non-silence phones as integers, separated by colons, e.g. 4:5:6:7:8:9");
parser.add_argument("silence_phones", type=str,
                    help="List of silence phones as integers, separated by colons, e.g. 1:2:3");
parser.add_argument("phone_list", type=str, help="file containing all phones and their corresponding number.");

args = parser.parse_args()

silence_phones = [ int(x) for x in args.silence_phones.split(":") ]
nonsilence_phones = [ int(x) for x in args.nonsilence_phones.split(":") ]
all_phones = silence_phones +  nonsilence_phones

cj5_phones = []
with open(args.phone_list) as f:
    for line in f:
        line = line.strip()
        phone = line.split(' ')[0]
        if "cj5" in phone:
            cj5_phones.append(int(line.split(' ')[1]))
# For nonsilence phones that are not punctuations
print("<Topology>")
print("<TopologyEntry>")
print("<ForPhones>")
print(" ".join([str(x) for x in nonsilence_phones if x not in cj5_phones]))
print("</ForPhones>")
for x in range(0, args.num_nonsil_states):
    xp1 = x + 1
    print("<State> " + str(x) + " <PdfClass> " + str(x) + " <Transition> " + str(x) + " 0.75 <Transition> " + str(xp1) + " 0.25 </State>")
print("<State> " + str(args.num_nonsil_states) + " </State>")
print("</TopologyEntry>")

# For nonsilence phones that are cj5
print("<TopologyEntry>")
print("<ForPhones>")
print(" ".join([str(x) for x in nonsilence_phones if x in cj5_phones]))
print("</ForPhones>")
for x in range(0, args.num_cj5_states):
    xp1 = x + 1
    print("<State> " + str(x) + " <PdfClass> " + str(x) + " <Transition> " + str(x) + " 0.75 <Transition> " + str(xp1) + " 0.25 </State>")
print("<State> " + str(args.num_cj5_states) + " </State>")
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
        state_str = state_str + "<Transition> " + str(x) + " " + str(transp) + " "
    state_str = state_str + "</State>"
    print(state_str)

    for x in range(1, (args.num_sil_states - 1)):
        state_str = "<State> " + str(x) + " <PdfClass> " + str(x) + " "
        for y in range(1, args.num_sil_states):
            state_str = state_str + "<Transition> " + str(y) + " " + str(transp) + " "
        state_str = state_str + "</State>"
        print(state_str)
    second_last = args.num_sil_states - 1
    print("<State> " + str(second_last) + " <PdfClass> " + str(second_last) + " <Transition> " + str(second_last) + " 0.75 <Transition> " + str(args.num_sil_states) + " 0.25 </State>")
    print("<State> " + str(args.num_sil_states) + " </State>")
else:
    print("<State> 0 <PdfClass> 0 <Transition> 0 0.75 <Transition> 1 0.25 </State>")
    print("<State> " + str(args.num_sil_states) + " </State>")
print("</TopologyEntry>")
print("</Topology>")
