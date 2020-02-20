#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import argparse
import os

import kaldi


def get_args():
    parser = argparse.ArgumentParser(description='convert text to labels')

    parser.add_argument('--lexicon-filename', dest='lexicon_filename', type=str)
    parser.add_argument('--tokens-filename', dest='tokens_filename', type=str)
    parser.add_argument('--dir', help='input/output dir', type=str)

    args = parser.parse_args()

    assert os.path.isfile(args.lexicon_filename)
    assert os.path.isfile(args.tokens_filename)
    assert os.path.isfile(os.path.join(args.dir, 'text'))

    return args


def read_lexicon(filename):
    '''
    Returns:
        a dict whose keys are words and values are phones.
    '''
    lexicon = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word_phones = line.split()
            assert len(word_phones) >= 2

            word = word_phones[0]
            phones = word_phones[1:]

            if word not in lexicon:
                # if there are multiple pronunciations for a word,
                # we choose only the first one and drop other alternatives
                lexicon[word] = phones

    return lexicon


def read_tokens(filename):
    '''
    Returns:
        a dict whose keys are phones and values are phone indices
    '''
    tokens = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            phone_index = line.split()
            assert len(phone_index) == 2

            phone = phone_index[0]
            index = int(phone_index[1])

            if phone == '<eps>':
                continue

            # decreased by one since we removed <eps> above
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
    '''
    Returns:
        a dict whose keys are utterance IDs and values are texts
    '''
    transcript = dict()

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            utt_text = line.split()
            assert len(utt_text) >= 2

            utt = utt_text[0]
            text = utt_text[1:]

            assert utt not in transcript
            transcript[utt] = text

    return transcript


def phones_to_indices(phone_list, tokens):
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

    for utt, text in transcript.items():
        labels = []
        for t in text:
            # TODO(fangjun): add support for OOV.
            phones = lexicon[t]

            indices = phones_to_indices(phones, tokens)

            labels.extend(indices)

        assert utt not in transcript_labels

        transcript_labels[utt] = labels

    wspecifier = 'ark,scp:{dir}/labels.ark,{dir}/labels.scp'.format(
        dir=args.dir)

    writer = kaldi.IntVectorWriter(wspecifier)

    for utt, labels in transcript_labels.items():
        writer.Write(utt, labels)

    writer.Close()

    print('Generated label file {}/labels.scp successfully'.format(args.dir))


if __name__ == '__main__':
    main()
