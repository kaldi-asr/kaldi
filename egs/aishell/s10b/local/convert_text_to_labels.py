#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

# This program converts a transcript file `text` to labels
# used in CTC training.
#
# For example, if we have
#
# the lexicon file `lexicon.txt`
#
# foo f o o
# bar b a r
#
# the phone symbol table `tokens.txt`
#
# <eps> 0
# <blk> 1
# a 2
# b 3
# f 4
# o 5
# r 6
#
# and the transcript file `text`
#
# utt1 foo bar bar
# utt2 bar
#
# Given the above three inputs, this program generates a
# file `labels.ark` containing
#
# utt1 3 4 4 2 1 5 2 1 5
# utt2 2 1 5
#
# where
# - `3 4 4` is from `(4-1) (5-1) (5-1)`, which is from the indices of `f o o`
# - `2 1 5` is from `(3-1) (2-1) (6-1)`, which is from the indices of `b a r`
#
# Note that 1 is subtracted from here since `<eps>` exists only in FSTs
# and the neural network considers index `0` as `<blk>`, Therefore, the integer
# value of every symbol is shifted downwards by 1.

import argparse
import os

import kaldi


def get_args():
    parser = argparse.ArgumentParser(description='''
Convert transcript to labels.

It takes the following inputs:

- lexicon.txt, the lexicon file
- tokens.txt, the phone symbol table
- dir, a directory containing the transcript file `text`

It generates `lables.scp` and `labels.ark` in the provided `dir`.

Usage:
    python3 ./local/convert_text_to_labels.py \
            --lexicon-filename data/lang/lexicon.txt \
            --tokens-filename data/lang/tokens.txt \
            --dir data/train

    It will generate data/train/labels.scp and data/train/labels.ark.
        ''')

    parser.add_argument('--lexicon-filename',
                        dest='lexicon_filename',
                        type=str,
                        help='filename for lexicon.txt')

    parser.add_argument('--tokens-filename',
                        dest='tokens_filename',
                        type=str,
                        help='filename for the phone symbol table tokens.txt')

    parser.add_argument('--dir',
                        type=str,
                        help='''the dir containing the transcript text;
        it will contain the generated labels.scp and labels.ark''')

    args = parser.parse_args()

    assert os.path.isfile(args.lexicon_filename)
    assert os.path.isfile(args.tokens_filename)
    assert os.path.isfile(os.path.join(args.dir, 'text'))

    return args


def read_lexicon(filename):
    '''Read lexicon.txt and save it into a Python dict.

    Args:
        filename: filename of lexicon.txt.

                  Every line in lexicon.txt has the following format:

                    word phone1 phone2 phone3 ... phoneN

                  That is, fields are separated by spaces. The first
                  field is the word and the remaining fields are the
                  phones indicating the pronunciation of the word.

    Returns:
        a dict whose keys are words and values are phones.
    '''
    lexicon = dict()

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # line contains:
            # word phone1 phone2 phone3 ... phoneN
            word_phones = line.split()

            # It should have at least two fields:
            # the first one is the word and
            # the second one is the pronunciation
            assert len(word_phones) >= 2

            word = word_phones[0]
            phones = word_phones[1:]

            if word not in lexicon:
                # if there are multiple pronunciations for a word,
                # we choose only the first one and drop other alternatives
                lexicon[word] = phones

    return lexicon


def read_tokens(filename):
    '''Read phone symbol table tokens.txt and save it into a Python dict.

    Note that we remove the symbol `<eps>` and shift every symbol index
    downwards by 1.

    Args:
        filename: filename of the phone symbol table tokens.txt.

                  Two integer values have specific meanings in the symbol
                  table. The first one is 0, which is reserved for `<eps>`.
                  And the second one is 1, which is reserved for the
                  blank symbol `<blk>`.
                  Other integer values do NOT have specific meanings.

    Returns:
        a dict whose keys are phones and values are phone indices
    '''
    tokens = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # line has the format: phone index
            phone_index = line.split()

            # it should have two fields:
            # the first field is the phone
            # and the second field is its index
            assert len(phone_index) == 2

            phone = phone_index[0]
            index = int(phone_index[1])

            if phone == '<eps>':
                # <eps> appears only in the FSTs.
                continue

            # decreased by one since we removed <eps> above
            # and every symbol index is shifted downwards by 1
            index -= 1

            assert phone not in tokens

            tokens[phone] = index

    assert '<blk>' in tokens

    # WARNING(fangjun): we assume that the blank symbol has index 0
    # in the neural network output.
    # Do NOT confuse it with `<eps>` in fst.
    assert tokens['<blk>'] == 0

    return tokens


def read_text(filename):
    '''Read transcript file `text` and save it into a Python dict.

    Args:
        filename: filename of the transcript file `text`.

    Returns:
        a dict whose keys are utterance IDs and values are texts
    '''
    transcript = dict()

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # line has the format: uttid word1 word2 word3 ... wordN
            uttid_text = line.split()

            # it should have at least 2 fields:
            # the first field is the utterance id;
            # the remaining fields are the words of the utterance
            assert len(uttid_text) >= 2

            uttid = uttid_text[0]
            text = uttid_text[1:]

            assert uttid not in transcript
            transcript[uttid] = text

    return transcript


def phones_to_indices(phone_list, tokens):
    '''Convert a list of phones to a list of indices via a phone symbol table.

    Args:
        phone_list: a list of phones
        tokens: a dict representing a phone symbol table.

    Returns:
        Return a list of indices corresponding to the given phones
    '''
    index_list = []

    for phone in phone_list:
        assert phone in tokens

        index = tokens[phone]
        index_list.append(index)

    return index_list


def main():
    args = get_args()

    lexicon = read_lexicon(args.lexicon_filename)

    tokens = read_tokens(args.tokens_filename)

    transcript = read_text(os.path.join(args.dir, 'text'))

    transcript_labels = dict()

    for uttid, text in transcript.items():
        labels = []
        for word in text:
            # TODO(fangjun): add support for OOV.
            phones = lexicon[word]

            indices = phones_to_indices(phones, tokens)

            labels.extend(indices)

        assert uttid not in transcript_labels

        transcript_labels[uttid] = labels

    wspecifier = 'ark,scp:{dir}/labels.ark,{dir}/labels.scp'.format(
        dir=args.dir)

    writer = kaldi.IntVectorWriter(wspecifier)

    for uttid, labels in transcript_labels.items():
        writer.Write(uttid, labels)

    writer.Close()

    print('Generated label file {}/labels.scp successfully'.format(args.dir))


if __name__ == '__main__':
    main()
