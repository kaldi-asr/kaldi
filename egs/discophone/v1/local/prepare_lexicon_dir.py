#!/usr/bin/env python3

import argparse
from itertools import chain
from pathlib import Path


special_word_to_special_phone = {
    '<hes>': '<unk>',
    '<noise>': '<noise>',
    '<silence>': '<silence>',
    '<unk>': '<unk>',
    '<v-noise>': '<noise>'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lexiconp')
    parser.add_argument('output_dir')
    parser.add_argument('--phone-tokens', action='store_true',
                        help='Will output phone tokens instead of phones.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    words, transcripts = [], []
    with open(args.lexiconp) as f:
        for line in (line.strip().split(maxsplit=2) for line in f):
            words.append(line[0])
            if len(line) > 2:
                transcripts.append(line[2])
            else:
                transcripts.append('')

    unique_phones = set(chain.from_iterable(t.split() for t in transcripts))
    unique_phone_tokens = set(chain.from_iterable(p for p in unique_phones))

    items = unique_phone_tokens if args.phone_tokens else unique_phones
    transcripts = [' '.join(t.replace(' ', '')) for t in transcripts] if args.phone_tokens else transcripts

    with open(output_dir / 'lexicon.txt', 'w') as f:
        for word, transcript in zip(words, transcripts):
            if word.startswith('<'):
                print(f'{word} {special_word_to_special_phone.get(word, "<unk>")}', file=f)
            else:
                print(f'{word} {transcript if transcript else "<unk>"}', file=f)

    with open(output_dir / 'silence_phones.txt', 'w') as f:
        for p in sorted(set(special_word_to_special_phone.values())):
            print(p, file=f)

    with open(output_dir / 'optional_silence.txt', 'w') as f:
        print('<silence>', file=f)

    with open(output_dir / 'nonsilence_phones.txt', 'w') as f:
        for p in sorted(items):
            print(p, file=f)

    (output_dir / 'extra_questions.txt').touch(exist_ok=True)


if __name__ == '__main__':
    main()
