#!/usr/bin/env python3
# coding: utf-8
import sys
import re

WORDLIST = dict()
IV_WORDS = dict()
OOV_WORDS = dict()
UNK = '<UNK>'
REPLACE_UNKS = True


def case_normalize(w):
    if w.startswith('~'):
        return w.upper()
    else:
        return w.lower()


def process_line(line):
    global WORDLIST
    tmp = re.sub(r'extreme\s+background', 'extreme_background', line)
    tmp = re.sub(r'foreign\s+lang=', 'foreign_lang=', tmp)
    tmp = re.sub(r'\)\)([^\s])', ')) \1', tmp)
    tmp = re.sub(r'[.,!?]', ' ', tmp)
    tmp = re.sub(r' -- ', ' ', tmp)
    tmp = re.sub(r' --$', '', tmp)
    x = re.split(r'\s+', tmp)
    old_x = x
    x = list()

    w = old_x.pop(0)
    while old_x:
        if w.startswith(r'(('):
            while old_x and not w.endswith('))'):
                w2 = old_x.pop(0)
                w += ' ' + w2
                #print(w, file=sys.stderr)
            x.append(w)
            if old_x:
                w = old_x.pop(0)
        elif w.startswith(r'<'):
            #this is very simplified and assumes we will not get a starting tag
            #alone
            while old_x and not w.endswith('>'):
                w2 = old_x.pop(0)
                w += ' ' + w2
            x.append(w)
            if old_x:
                w = old_x.pop(0)
        elif w.endswith(r'))'):
            print('error ' + w, file=sys.stderr)
            if old_x:
                w = old_x.pop(0)
        else:
            x.append(w)
            if old_x:
                w = old_x.pop(0)

    if not x:
        return None
    if len(x) == 1 and x[0] in ('<background>', '<extreme_background>'):
        return None

    out_x = list()
    for w in x:
        w = case_normalize(w)
        if w in WORDLIST:
            IV_WORDS[w] = 1 + IV_WORDS.get(w, 0)
            out_x.append(w)
        else:
            OOV_WORDS[w] = 1 + OOV_WORDS.get(w, 0)
            if REPLACE_UNKS:
                out_x.append(UNK)
            else:
                out_x.append(w)

    return ' '.join(out_x)


def read_lexicon_words(lexicon):
    with open(lexicon, 'r', encoding='utf-8') as f:
        for line in f:
            line = re.sub(r'(?s)\s.*', '', line)
            WORDLIST[line] = 1


def main(lexicon, transcript_file, output_file):
    read_lexicon_words(lexicon)

    f_out = open(output_file, 'w', encoding='utf-8')

    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = re.split(r'\s+', line, 5)
            cleaned_line = process_line(line[5])
            if cleaned_line:
                line[5] = (cleaned_line)
                print(' '.join(line), file=f_out)

    s = sorted(((v, k) for k, v in OOV_WORDS.items()), reverse=True)
    for v, k in s:
        print('{} {}'.format(v,k))
        #print(f'{v} {k}')
    #s = sorted(((v, k) for k, v in WORDLIST.items()), reverse=True)
    #for v, k in s:
    #    print(f'{k} : {v}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-unk-replace',
                        action='store_true',
                        help='Do not replace the unknown words by <UNK> ')
    parser.add_argument('--unk', type=str, default=UNK,
                        help='Do not replace the unknown words by <UNK> ')

    parser.add_argument('lexicon', type=str,
                        help='lexicon file')
    parser.add_argument('input', type=str,
                        help='input transcript')
    parser.add_argument('output', type=str,
                        help='output transcript')
    args = parser.parse_args()

    UNK = args.unk
    REPLACE_UNKS = not args.no_unk_replace
    main(args.lexicon, args.input, args.output)
