#!/usr/bin/env python

# Copyright     2017  Ashish Arora

import argparse
import sys

parser = argparse.ArgumentParser(description="""uses phones to convert unk to word""")
parser.add_argument('phones', type=str, help='phones and phonesID')
parser.add_argument('words', type=str, help='word and wordID')
parser.add_argument('unk', type=str, default='-', help='location of unk file')
parser.add_argument('--input-ark', type=str, default='-', help='where to read the input data')
parser.add_argument('--out-ark', type=str, default='-', help='where to write the output data')
args = parser.parse_args()
### main ###
phone_fh = open(args.phones, 'r')
word_fh = open(args.words, 'r')
unk_fh = open(args.unk,'r')
if args.input_ark == '-':
    input_fh = sys.stdin
else:
    input_fh = open(args.input_ark,'r')
if args.out_ark == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.out_ark,'wb')

phone_dict = dict()# stores phoneID and phone mapping
phone_data_vect = phone_fh.read().strip().split("\n")
for key_val in phone_data_vect:
  key_val = key_val.split(" ")
  phone_dict[key_val[1]] = key_val[0]
word_dict = dict()
word_data_vect = word_fh.read().strip().split("\n")
for key_val in word_data_vect:
  key_val = key_val.split(" ")
  word_dict[key_val[1]] = key_val[0]
unk_val = unk_fh.read().strip().split(" ")[0]

utt_word_dict = dict()
utt_phone_dict = dict()# stores utteranceID and phoneID
unk_word_dict = dict()
count=0
for line in input_fh:
  line_vect = line.strip().split("\t")
  if len(line_vect) < 6:
    print "IndexError"
    print line_vect
    continue
  uttID = line_vect[0]
  word = line_vect[4]
  phones = line_vect[5]
  if uttID in utt_word_dict.keys():
    utt_word_dict[uttID][count] = word
    utt_phone_dict[uttID][count] = phones
  else:
    count = 0
    utt_word_dict[uttID] = dict()
    utt_phone_dict[uttID] = dict()
    utt_word_dict[uttID][count] = word
    utt_phone_dict[uttID][count] = phones
  if word == unk_val: # get character sequence for unk
    phone_key_vect = phones.split(" ")
    phone_val_vect = list()
    for pkey in phone_key_vect:
      phone_val_vect.append(phone_dict[pkey])
    phone_2_word = list()
    for phone_val in phone_val_vect:
      phone_2_word.append(phone_val.split('_')[0])
    phone_2_word = ''.join(phone_2_word)
    utt_word_dict[uttID][count] = phone_2_word
  else:
    if word == '0':
      word_val = ' '
    else:
      word_val = word_dict[word]
    utt_word_dict[uttID][count] = word_val
  count += 1

transcription = ""
for key in sorted(utt_word_dict.iterkeys()):
  transcription = key
  for index in sorted(utt_word_dict[key].iterkeys()):
    value = utt_word_dict[key][index]
    transcription = transcription + " " + value
  out_fh.write(transcription + '\n')
