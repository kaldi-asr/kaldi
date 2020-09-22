#!/usr/bin/env python3

import argparse
from itertools import chain
from pathlib import Path
from typing import Dict

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
    parser.add_argument('text')
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
                transcripts.append(' '.join(tok for tok in line[2].replace('.', '').replace('#', '').split() if tok))
            else:
                transcripts.append('')

    unique_phones = set(chain.from_iterable(t.split() for t in transcripts))
    unique_phone_tokens = set(chain.from_iterable(p for p in unique_phones))

    items = unique_phone_tokens if args.phone_tokens else unique_phones
    transcripts = [' '.join(t.replace(' ', '')) for t in transcripts] if args.phone_tokens else [t.strip() for t in
                                                                                                 transcripts]

    # Maps each word to a single pronunciation variant; ignores pronunciation alternatives,
    # becuase it's not obvious how to select them for LM training text generation;
    # random sampling weighted by pron prob likely won't improve that much...
    word_lexicon: Dict[str, str] = {}
    for word, phone_trs in zip(words, transcripts):
        if word in word_lexicon:
            continue
        word_lexicon[word] = phone_trs

    with open(output_dir / 'phones_text', 'w') as fout, open(args.text) as fin:
        for line in fin:
            utt_id, *words = line.strip().split()
            # A heuristic - if the dev word is not in the lexicon,
            # generate as many <unk> phones as the number of characters.
            print(' '.join(word_lexicon.get(w, ' '.join(['<unk>'] * len(w))) for w in words), file=fout)

    with open(output_dir / 'lexicon.txt', 'w') as f_lex, \
            open(output_dir / 'lexiconp.txt', 'w') as f_lexp:
        for phone_unit in items:
            print(f'{phone_unit} {phone_unit}', file=f_lex)
            print(f'{phone_unit} 1 {phone_unit}', file=f_lexp)

    with open(output_dir / 'silence_phones.txt', 'w') as f:
        for p in sorted(set(special_word_to_special_phone.values()) - {'<unk>'}):
            print(p, file=f)

    with open(output_dir / 'optional_silence.txt', 'w') as f:
        print('<silence>', file=f)

    with open(output_dir / 'nonsilence_phones.txt', 'w') as f:
        for p in sorted(items):
            print(p, file=f)
        print('<unk>', file=f)

    (output_dir / 'extra_questions.txt').touch(exist_ok=True)


if __name__ == '__main__':
    main()
