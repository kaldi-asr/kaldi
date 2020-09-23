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
    parser.add_argument('--phones', action='store_true', help='Will output phones (default).')
    parser.add_argument('--phone-tokens', action='store_true',
                        help='Will output phone tokens instead of phones.')
    args = parser.parse_args()
    assert not (args.phones and args.phone_tokens), "--phones and --phone-tokens options are exclusive!"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    words, transcripts = [], []
    with open(args.lexiconp) as f:
        for line in (line.strip().split(maxsplit=2) for line in f):
            words.append(line[0])
            if len(line) > 2:
                transcripts.append(' '.join(tok for tok in line[2].replace('.', '').replace('#', '').split() if tok))
            else:
                transcripts.append('')

    unique_phones = set(chain.from_iterable(t.split() for t in transcripts))
    unique_phone_tokens = set(chain.from_iterable(p for p in unique_phones))

    items = unique_phone_tokens if args.phone_tokens else unique_phones
    transcripts = [' '.join(t.replace(' ', '')) for t in transcripts] if args.phone_tokens else [t.strip() for t in
                                                                                                 transcripts]

    with open(output_dir / 'lexicon.txt', 'w') as f_lex, \
            open(output_dir / 'lexiconp.txt', 'w') as f_lexp:
        if '<unk>' not in words:
            print('<unk> <unk>', file=f_lex)
            print('<unk> 1 <unk>', file=f_lexp)
        for word, transcript in zip(words, transcripts):
            if word.startswith('<'):
                print(f'{word} {special_word_to_special_phone.get(word, "<unk>")}', file=f_lex)
                print(f'{word} 1 {special_word_to_special_phone.get(word, "<unk>")}', file=f_lexp)
            else:
                if transcript:
                    print(f'{word} {transcript}', file=f_lex)
                    print(f'{word} 1 {transcript}', file=f_lexp)

    with open(output_dir / 'silence_phones.txt', 'w') as f:
        for p in sorted(set(special_word_to_special_phone.values()) - {'<unk>'}):
            print(p, file=f)

    with open(output_dir / 'optional_silence.txt', 'w') as f:
        print('<silence>', file=f)

    with open(output_dir / 'nonsilence_phones.txt', 'w') as f:
        print('<unk>', file=f)
        for p in sorted(items):
            print(p, file=f)

    (output_dir / 'extra_questions.txt').touch(exist_ok=True)


if __name__ == '__main__':
    main()
