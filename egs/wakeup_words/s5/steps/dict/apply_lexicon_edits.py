#!/usr/bin/env python

# Copyright 2016  Xiaohui Zhang
# Apache 2.0.

from __future__ import print_function
import argparse
import sys

def GetArgs():
    parser = argparse.ArgumentParser(description = "Apply an lexicon edits file (output from steps/dict/select_prons_bayesian.py)to an input lexicon"
                                     "to produce a learned lexicon.",
                                     epilog = "See steps/dict/learn_lexicon_greedy.sh for example")

    parser.add_argument("in_lexicon", metavar='<in-lexicon>', type = str,
                        help = "Input lexicon. Each line must be <word> <phones>.")
    parser.add_argument("lexicon_edits_file", metavar='<lexicon-edits-file>', type = str,
                        help = "Input lexicon edits file containing human-readable & editable"
                               "pronounciation info.  The info for each word is like:"
                         "------------ an 4086.0 --------------"
                         "R  | Y |  2401.6 |  AH N"
                         "R  | Y |  640.8 |  AE N"
                         "P  | Y |  1035.5 |  IH N"
                         "R(ef), P(hone-decoding) represents the pronunciation source"
                         "Y/N means the recommended decision of including this pron or not"
                         "and the numbers are soft counts accumulated from lattice-align-word outputs. See steps/dict/select_prons_bayesian.py for more details.")
    parser.add_argument("out_lexicon", metavar='<out-lexicon>', type = str,
                        help = "Output lexicon to this file.")

    print (' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if args.in_lexicon == "-":
        args.in_lexicon = sys.stdin
    else:
        args.in_lexicon_handle = open(args.in_lexicon)
    args.lexicon_edits_file_handle = open(args.lexicon_edits_file)

    if args.out_lexicon == "-":
        args.out_lexicon_handle = sys.stdout
    else:
        args.out_lexicon_handle = open(args.out_lexicon, "w")

    return args

def ReadLexicon(lexicon_file_handle):
    lexicon = set()
    if lexicon_file_handle:
        for line in lexicon_file_handle.readlines():
            splits = line.strip().split()
            if len(splits) == 0:
                continue
            if len(splits) < 2:
                raise Exception('Invalid format of line ' + line
                                    + ' in lexicon file.')
            word = splits[0]
            phones = ' '.join(splits[1:])
            lexicon.add((word, phones))
    return lexicon

def ApplyLexiconEdits(lexicon, lexicon_edits_file_handle):
    if lexicon_edits_file_handle:
        for line in lexicon_edits_file_handle.readlines():
            # skip all commented lines
            if line.startswith('#'):
                continue
            # read a word from a line like "---- MICROPHONES 200.0 ----".
            if line.startswith('---'):
                splits = line.strip().strip('-').strip().split()
                if len(splits) != 2:
                    print(splits, file=sys.stderr)
                    raise Exception('Invalid format of line ' + line
                                        + ' in lexicon edits file.')
                word = splits[0].strip()
            else:
            # parse the pron and decision 'Y/N' of accepting the pron or not,
            # from a line like: 'P  | Y |  42.0 |  M AY K R AH F OW N Z'
                splits = line.split('|')
                if len(splits) != 4:
                    raise Exception('Invalid format of line ' + line
                                        + ' in lexicon edits file.')
                pron = splits[3].strip()
                if splits[1].strip() == 'Y':
                    lexicon.add((word, pron))
                elif splits[1].strip() == 'N':
                    lexicon.discard((word, pron))
                else:
                    raise Exception('Invalid format of line ' + line
                                        + ' in lexicon edits file.')
    return lexicon


def WriteLexicon(lexicon, out_lexicon_handle):
    for word, pron in lexicon:
        print('{0} {1}'.format(word, pron), file=out_lexicon_handle)
    out_lexicon_handle.close()

def Main():
    args = GetArgs()
    lexicon = ReadLexicon(args.in_lexicon_handle)
    ApplyLexiconEdits(lexicon, args.lexicon_edits_file_handle)
    WriteLexicon(lexicon, args.out_lexicon_handle)

if __name__ == "__main__":
    Main()
