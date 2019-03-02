#!/usr/bin/env python3

# Copyright    2018 Hossein Hadian
# Apache 2.0

import argparse
from os.path import join
import sys
import copy
import random

parser = argparse.ArgumentParser(description="""This script reads
    sequences of phone ids from std input and counts mono/biphone stats
    and writes the results to std out. The output can be used with
    gmm-init-biphone to create a better tree. The first part of the
    outupt is biphone counts with this format for each line:
    <phone-id> <phone-id> <count>
    and the second part of the output is monophone counts with the
    following format:
    <phone-id> <count>""")
parser.add_argument('langdir', type=str)
parser.add_argument('--shared-phones', type=str, choices=['true','false'],
                    default='true',
                    help="If true, stats will be collected for shared phones.")

args = parser.parse_args()
args.shared_phones = True if args.shared_phones == 'true' else False

# Read phone sets
phone_sets = []
phones = []
phone_to_shard_phone = {}
phone_to_shard_phone[0] = 0  # The no-left-context case
with open(join(args.langdir, 'phones/sets.int'), 'r', encoding='latin-1') as f:
    for line in f:
        phone_set = line.strip().split()
        phone_sets.append(phone_set)
        for phone in phone_set:
            phones.append(phone)
            phone_to_shard_phone[phone] = phone_set[0]

print('Loaded {} phone-sets containing {} phones.'.format(len(phone_sets),
                                                          len(phones)),
      file=sys.stderr)

biphone_counts = {}
mono_counts = {}
for line in sys.stdin:
    line = line.strip().split()
    key = line[0]
    line_phones = line[1:]
    for pair in zip([0] + line_phones, line_phones):  # 0 is for the no left-context case
        if args.shared_phones:
            pair = (phone_to_shard_phone[pair[0]], phone_to_shard_phone[pair[1]])
        if pair not in biphone_counts:
            biphone_counts[pair] = 0
        biphone_counts[pair] += 1
        mono_counts[pair[1]] = 1 if pair[1] not in mono_counts else mono_counts[pair[1]] + 1

for phone1 in [0] + phones:
    for phone2 in phones:
        pair = (phone1, phone2)
        shared_pair = ((phone_to_shard_phone[pair[0]], phone_to_shard_phone[pair[1]])
                       if args.shared_phones else pair)
        count = biphone_counts[shared_pair] if shared_pair in biphone_counts else 0
        if count != 0:
            print('{} {} {}'.format(pair[0], pair[1], count))
for phone in phones:
    shared = phone_to_shard_phone[phone] if args.shared_phones else phone
    count = mono_counts[shared] if shared in mono_counts else 0
    if count != 0:
        print('{} {}'.format(phone, count))
