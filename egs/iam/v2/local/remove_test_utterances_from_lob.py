#!/usr/bin/env python3
# Copyright   2018 Ashish Arora

import argparse
import os
import numpy as np
import sys
import re

parser = argparse.ArgumentParser(description="""Removes dev/test set lines
                                                from the LOB corpus. Reads the
                                                corpus from stdin, and writes it to stdout.""")
parser.add_argument('dev_text', type=str,
                    help='dev transcription location.')
parser.add_argument('test_text', type=str,
                    help='test transcription location.')
args = parser.parse_args()

def remove_punctuations(transcript):
    char_list = []
    for char in transcript:
        if char.isdigit() or char == '+' or char == '~' or char == '?':
            continue
        if char == '#' or char == '=' or char == '-' or char == '!':
            continue
        if char == ',' or char == '.' or char == ')' or char == '\'':
            continue
        if char == '(' or char == ':' or char == ';' or char == '"':
            continue
        if char == '*':
            continue
        char_list.append(char)
    return char_list


def remove_special_words(words):
    word_list = []
    for word in words:
        if word == '<SIC>' or word == '#':
            continue
        word_list.append(word)
    return word_list


# process and add dev/eval transcript in a list
# remove special words, punctuations, spaces between words
# lowercase the characters
def read_utterances(text_file_path):
    with open(text_file_path, 'rt') as in_file:
        for line in in_file:
            words = line.strip().split()
            words_wo_sw = remove_special_words(words)
            transcript = ''.join(words_wo_sw[1:])
            transcript = transcript.lower()
            trans_wo_punct = remove_punctuations(transcript)
            transcript = ''.join(trans_wo_punct)
            utterance_dict[words_wo_sw[0]] = transcript


### main ###

# read utterances and add it to utterance_dict
utterance_dict = dict()
read_utterances(args.dev_text)
read_utterances(args.test_text)

# read corpus and add it to below lists
corpus_text_lowercase_wo_sc = list()
corpus_text_wo_sc = list()
original_corpus_text = list()
for line in sys.stdin:
    original_corpus_text.append(line)
    words = line.strip().split()
    words_wo_sw = remove_special_words(words)

    transcript = ''.join(words_wo_sw)
    transcript = transcript.lower()
    trans_wo_punct = remove_punctuations(transcript)
    transcript = ''.join(trans_wo_punct)
    corpus_text_lowercase_wo_sc.append(transcript)

    transcript = ''.join(words_wo_sw)
    trans_wo_punct = remove_punctuations(transcript)
    transcript = ''.join(trans_wo_punct)
    corpus_text_wo_sc.append(transcript)

# find majority of utterances below
# for utterances which were not found
# add them to remaining_utterances
row_to_keep = [True for i in range(len(original_corpus_text))]
remaining_utterances = dict()
for line_id, line_to_find in utterance_dict.items():
    found_line = False
    # avoiding very small utterance, it causes removing
    # complete lob text
    if len(line_to_find) < 10:
        remaining_utterances[line_id] = line_to_find
    else:
        for i in range(1, (len(corpus_text_lowercase_wo_sc) - 2)):
            # Combine 3 consecutive lines of the corpus into a single line
            prev_words = corpus_text_lowercase_wo_sc[i - 1].strip()
            curr_words = corpus_text_lowercase_wo_sc[i].strip()
            next_words = corpus_text_lowercase_wo_sc[i + 1].strip()
            new_line = prev_words + curr_words + next_words
            transcript = ''.join(new_line)
            if line_to_find in transcript:
                found_line = True
                row_to_keep[i-1] = False
                row_to_keep[i] = False
                row_to_keep[i+1] = False
    if not found_line:
        remaining_utterances[line_id] = line_to_find

# removing long utterances not found above
row_to_keep[87530] = False; row_to_keep[87531] = False; row_to_keep[87532] = False;
row_to_keep[31724] = False; row_to_keep[31725] = False; row_to_keep[31726] = False;
row_to_keep[16704] = False; row_to_keep[16705] = False; row_to_keep[16706] = False;
row_to_keep[94181] = False; row_to_keep[94182] = False; row_to_keep[94183] = False;
row_to_keep[20171] = False; row_to_keep[20172] = False; row_to_keep[20173] = False;
row_to_keep[16734] = False; row_to_keep[16733] = False; row_to_keep[16732] = False;
row_to_keep[20576] = False; row_to_keep[20577] = False; row_to_keep[20578] = False;
row_to_keep[31715] = False; row_to_keep[31716] = False; row_to_keep[31717] = False;
row_to_keep[31808] = False; row_to_keep[31809] = False; row_to_keep[31810] = False;
row_to_keep[31822] = False; row_to_keep[31823] = False; row_to_keep[31824] = False;
row_to_keep[88791] = False; row_to_keep[88792] = False; row_to_keep[88793] = False;
row_to_keep[31745] = False; row_to_keep[31746] = False; row_to_keep[31825] = False;
row_to_keep[94256] = False; row_to_keep[94257] = False; row_to_keep[88794] = False;
row_to_keep[88665] = False; row_to_keep[17093] = False; row_to_keep[17094] = False;
row_to_keep[20586] = False; row_to_keep[87228] = False; row_to_keep[87229] = False;
row_to_keep[16744] = False; row_to_keep[87905] = False; row_to_keep[87906] = False;
row_to_keep[16669] = False; row_to_keep[16670] = False; row_to_keep[16719] = False;
row_to_keep[87515] = False; row_to_keep[20090] = False; row_to_keep[31748] = False;
for i in range(len(original_corpus_text)):
    transcript = original_corpus_text[i].strip()
    if row_to_keep[i]:
        print(transcript)

print('Sentences not removed from LOB: {}'.format(remaining_utterances), file=sys.stderr)
print('Total test+dev sentences: {}'.format(len(utterance_dict)), file=sys.stderr)
print('Number of sentences not removed from LOB: {}'. format(len(remaining_utterances)), file=sys.stderr)
print('LOB lines: Before: {}   After: {}'.format(len(original_corpus_text),
                                                 row_to_keep.count(True)), file=sys.stderr)
