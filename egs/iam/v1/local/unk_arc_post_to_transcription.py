#!/usr/bin/env python3

#Copyright      2017  Ashish Arora

""" This module will be used by scripts for open vocabulary setup.
 If the hypothesis transcription contains <unk>, then it will replace the 
 <unk> with the word predicted by <unk> model by concatenating phones decoded 
 from the unk-model. It is currently supported only for triphone setup.
 Args:
  phones: File name of a file that contains the phones.txt, (symbol-table for phones).
          phone and phoneID, Eg. a 217, phoneID of 'a' is 217. 
  words: File name of a file that contains the words.txt, (symbol-table for words). 
         word and wordID. Eg. ACCOUNTANCY 234, wordID of 'ACCOUNTANCY' is 234.
  unk: ID of <unk>. Eg. 231.
  one-best-arc-post: A file in arc-post format, which is a list of timing info and posterior
               of arcs along the one-best path from the lattice.
               E.g. 506_m01-049-00 8 12  1 7722  282 272 288 231
                    <utterance-id> <start-frame> <num-frames> <posterior> <word> [<ali>] 
                    [<phone1> <phone2>...]
  output-text: File containing hypothesis transcription with <unk> recognized by the
               unk-model.
               E.g. A move to stop mr. gaitskell.
  
  Eg. local/unk_arc_post_to_transcription.py lang/phones.txt lang/words.txt 
      data/lang/oov.int
"""
import argparse
import os
import sys
parser = argparse.ArgumentParser(description="""uses phones to convert unk to word""")
parser.add_argument('phones', type=str, help='File name of a file that contains the'
                    'symbol-table for phones. Each line must be: <phone> <phoneID>')
parser.add_argument('words', type=str, help='File name of a file that contains the'
                    'symbol-table for words. Each line must be: <word> <word-id>')
parser.add_argument('unk', type=str, default='-', help='File name of a file that'
                    'contains the ID of <unk>. The content must be: <oov-id>, e.g. 231')
parser.add_argument('--one-best-arc-post', type=str, default='-', help='A file in arc-post'
                    'format, which is a list of timing info and posterior of arcs'
                    'along the one-best path from the lattice')
parser.add_argument('--output-text', type=str, default='-', help='File containing'
                    'hypothesis transcription with <unk> recognized by the unk-model')
args = parser.parse_args()

### main ###
phone_handle = open(args.phones, 'r', encoding='latin-1') # Create file handles 
word_handle = open(args.words, 'r', encoding='latin-1')
unk_handle = open(args.unk,'r', encoding='latin-1')
if args.one_best_arc_post == '-':
    arc_post_handle = sys.stdin
else:
    arc_post_handle = open(args.one_best_arc_post, 'r', encoding='latin-1')
if args.output_text == '-':
    output_text_handle = sys.stdout
else:
    output_text_handle = open(args.output_text, 'w', encoding='latin-1')

id2phone = dict() # Stores the mapping from phone_id (int) to phone (char)
phones_data = phone_handle.read().strip().split("\n")

for key_val in phones_data:
  key_val = key_val.split(" ")
  id2phone[key_val[1]] = key_val[0]

word_dict = dict()
word_data_vect = word_handle.read().strip().split("\n")

for key_val in word_data_vect:
  key_val = key_val.split(" ")
  word_dict[key_val[1]] = key_val[0]
unk_val = unk_handle.read().strip().split(" ")[0]

utt_word_dict = dict() # Dict of list, stores mapping from utteranceID(int) to words(str)
for line in arc_post_handle:
  line_vect = line.strip().split("\t")
  if len(line_vect) < 6: # Check for 1best-arc-post output
    print("Error: Bad line: '{}'   Expecting 6 fields. Skipping...".format(line),
          file=sys.stderr)
    continue
  utt_id = line_vect[0]
  word = line_vect[4]
  phones = line_vect[5]
  if utt_id not in list(utt_word_dict.keys()):
    utt_word_dict[utt_id] = list()

  if word == unk_val: # Get the 1best phone sequence given by the unk-model
    phone_id_seq = phones.split(" ")
    phone_seq = list()
    for pkey in phone_id_seq:
      phone_seq.append(id2phone[pkey]) # Convert the phone-id sequence to a phone sequence.
    phone_2_word = list()
    for phone_val in phone_seq:
      phone_2_word.append(phone_val.split('_')[0]) # Removing the world-position markers(e.g. _B)
    phone_2_word = ''.join(phone_2_word) # Concatnate phone sequence
    utt_word_dict[utt_id].append(phone_2_word) # Store word from unk-model
  else:
    if word == '0': # Store space/silence
      word_val = ' '
    else:
      word_val = word_dict[word]
    utt_word_dict[utt_id].append(word_val) # Store word from 1best-arc-post

transcription = "" # Output transcription
for utt_key in sorted(utt_word_dict.keys()):
  transcription = utt_key
  for word in utt_word_dict[utt_key]:
    transcription = transcription + " " + word
  output_text_handle.write(transcription + '\n')
