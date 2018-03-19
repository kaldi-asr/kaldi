#!/usr/bin/env python3

import argparse
import os
import numpy as np
import sys
import re

parser = argparse.ArgumentParser(description="""Removes dev/eval set lines
                                                from the corpus.""")
parser.add_argument('corpus_path', type=str,
                    help='Path to the corpus')
parser.add_argument('dev_path', type=str,
                    help='dev transcription location.')
parser.add_argument('eval_path', type=str,
                    help='eval transcription location.')
parser.add_argument('out_dir', type=str,
                    help='Where to write output file.')
args = parser.parse_args()

dataset_path = os.path.join(args.corpus_path, 'lob_1.txt')

text_file_path = os.path.join(args.dev_path,
                              'text')
text_file_path_1 = os.path.join(args.eval_path,
                                'text')

output_file = os.path.join(args.out_dir,
                           'lob.txt')


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


# process and add corpus sentences in a list
# remove special words, punctuations, spaces between words
# lowercase the characters
def read_corpus(dataset_path):
    with open(dataset_path, 'rt') as in_file:
        for line in in_file:
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


### main ###

# read utterances and add it to utterance_dict
utterance_dict = dict()
read_utterances(text_file_path)
read_utterances(text_file_path_1)

# read corpus and add it to below lists
corpus_text_lowercase_wo_sc = list()
corpus_text_wo_sc = list()
original_corpus_text = list()
read_corpus(dataset_path)

# find majority of utterances below
# for utterances which were not found
# add them to remaining_utterances
row_to_keep = [True for i in range(len(original_corpus_text))]
remaining_utterances = dict()
for line_id, line_to_find in utterance_dict.items():
    found_line = False
    for i in range(1, (len(corpus_text_lowercase_wo_sc) - 2)):
        # combine 3 consequtive lines of the corpus into single line
        prev_words = corpus_text_lowercase_wo_sc[i - 1].strip()
        curr_words = corpus_text_lowercase_wo_sc[i].strip()
        next_words = corpus_text_lowercase_wo_sc[i + 1].strip()
        new_line = prev_words + curr_words + next_words
        transcript = ''.join(new_line)
        if line_to_find in transcript:
            found_line = True
            row_to_keep[i] = False
    if not found_line:
        remaining_utterances[line_id] = line_to_find


text_fh = open(output_file, 'w')
for i in range(len(original_corpus_text)):
    transcript = original_corpus_text[i].strip()
    if row_to_keep[i]:
        text_fh.write(transcript + '\n')

print(remaining_utterances)
print(len(remaining_utterances))
print (len(original_corpus_text))
print (len(utterance_dict))
print (row_to_keep.count(False))

