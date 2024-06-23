#!/usr/bin/env python3
# Copyright 2021  Jiayu DU
#           2021  Xiaomi Corporation (Author: Yongqing Wang)

import sys, os

conversational_filler = ['UH', 'UHH', 'UM', 'EH', 'MM', 'HM', 'AH', 'HUH', 'HA', 'ER', 'OOF', 'HEE', 'HMM', 'ACH', 'EEE', 'EW']
unk_tags = ['<UNK>', '<unk>']
gigaspeech_punctuations = ['<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>']
gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']
non_scoring_words = conversational_filler + unk_tags + gigaspeech_punctuations + gigaspeech_garbage_utterance_tags


def asr_text_post_processing(text):
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace('-', ' ')

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    return ' '.join(remaining_words)


if __name__ == '__main__':
    for line in sys.stdin.readlines():
        if line.strip():
            cols = line.strip().split(maxsplit=1)
            key = cols[0]
            text = ''
            if len(cols) == 2:
                text = cols[1]
            print(F'{key} {asr_text_post_processing(text)}')
