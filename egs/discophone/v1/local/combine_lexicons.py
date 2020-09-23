#!/usr/bin/env python3
# Copyright 2020  Johns Hopkins University (Author: Piotr Żelasko)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lexicons', nargs='+')
    args = parser.parse_args()

    # Read individual lexicons
    lexicons = []
    for lex_path in args.lexicons:
        lex = defaultdict(list)
        with open(lex_path) as f:
            for line in f:
                word, *phones = line.strip().split()
                try:
                    score = float(phones[0])
                    phones = phones[1:]
                except:
                    # can't convert (not lexiconp) - the first phone is not a score
                    score = None
                if not phones:
                    continue
                lex[word].append((score, ' '.join(phones)))
        lexicons.append(lex)

    # Merge them
    merged = defaultdict(list)
    for lex in lexicons:
        for word, scores_and_prons in lex.items():
            prons = [p for _, p in merged[word]]
            for score, pron in scores_and_prons:
                if pron in prons:
                    continue
                merged[word].append((score, pron))

    # Sort and output
    for word in sorted(merged):
        for score, pron in merged[word]:
            if score is not None:
                print(word, score, pron)
            else:
                print(word, pron)


if __name__ == '__main__':
    main()
