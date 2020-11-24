#!/usr/bin/env python3
# This script sorts the utterance based in the error
# Ashish Arora
# License: Apache 2.0
from collections import OrderedDict
import sys, io
import string
infile = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

total_words = 0
total_error = 0
total_substitution = 0
total_insert = 0
total_deletion = 0
csid_dict = {}

for line in infile:
    line = line.strip().split(" ")
    utt_id = line[0]
    # get only csid lines
    if "csid" not in line[1]:
        continue

    correct = int(line[2])
    substitute = int(line[3])
    insert = int(line[4])
    delete = int(line[5])

    total_words = total_words + correct + substitute + delete
    total_error = total_error + substitute + delete + insert
    total_substitution = total_substitution + substitute
    total_insert = total_insert + insert
    total_deletion = total_deletion + delete
    csid_list = [correct, substitute, insert, delete]
    csid_dict[utt_id] = csid_list


print(total_words)
print(total_error)

precent_ins = total_insert * 100 / total_words
percent_del = total_deletion * 100 / total_words
percent_sub = total_substitution * 100 / total_words
wer = total_error * 100 / total_words

print(total_insert)
print(total_deletion)
print(total_substitution)
print("{:.2f}".format(wer))

for w in sorted(csid_dict, key=csid_dict.get, reverse=True):
    print(w, csid_dict[w])
