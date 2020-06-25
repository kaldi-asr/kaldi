#!/usr/bin/env python

# Copyright 2018 Xiaohui Zhang
# Apache 2.0.

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import argparse
import sys
import string

def GetArgs():
    parser = argparse.ArgumentParser(
        description = "The purpose of this script is to use a ctm and a vocab file"
        "to extract sub-utterances and a sub-segmentation. Extracted sub-utterances"
        "are all the strings of consecutive in-vocab words from the ctm"
        "surrounded by an out-of-vocab word at each end if present.",
        epilog = "e.g. steps/dict/internal/get_subsegments.py exp/tri3_lex_0.4_work/phonetic_decoding/word.ctm \\"
        "exp/tri3_lex_0.4_work/learn_vocab.txt exp/tri3_lex_0.4_work/resegmentation/subsegments \\"
        "exp/tri3_lex_0.4_work/resegmentation/text"
        "See steps/dict/learn_lexicon_greedy.sh for an example.")

    parser.add_argument("ctm", metavar='<ctm>', type = str,
                        help = "Input ctm file."
                        "each line must be <utt-id> <chanel> <start-time> <duration> <word>")
    parser.add_argument("vocab", metavar='<vocab>', type = str,
                        help = "Vocab file."
                        "each line must be <word>")
    parser.add_argument("subsegment", metavar='<subsegtment>', type = str,
                        help = "Subsegment file. Each line is in format:"
                        "<new-utt> <old-utt> <start-time-within-old-utt> <end-time-within-old-utt>")
    parser.add_argument("text", metavar='<text>', type = str,
                        help = "Text file. Each line is in format:"
                        " <new-utt> <word1> <word2> ... <wordN>.")
  
    print (' '.join(sys.argv), file = sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if args.ctm == "-":
        args.ctm_handle = sys.stdin
    else:
        args.ctm_handle = open(args.ctm)

    if args.vocab is not '':
        if args.vocab == "-":
            args.vocab_handle = sys.stdout
        else:
            args.vocab_handle = open(args.vocab)

    args.subsegment_handle = open(args.subsegment, 'w')
    args.text_handle = open(args.text, 'w')

    return args

def GetSubsegments(args, vocab):
    sub_utt = list()
    last_is_oov = False
    is_oov = False
    utt_id_last = None
    start_times = {}
    end_times = {}
    sub_utts = {}
    sub_utt_id = 1
    sub_utt_id_last = 1
    end_time_last = 0.0
    for line in args.ctm_handle:
        splits = line.strip().split()
        if len(splits) < 5:
            raise Exception("problematic line",line)

        utt_id = splits[0]
        start = float(splits[2])
        dur = float(splits[3])
        word = splits[4]
        if utt_id != utt_id_last:
            sub_utt_id = 1
            if len(sub_utt)>1:
                sub_utts[utt_id_last+'-'+str(sub_utt_id_last)] = (utt_id_last, sub_utt)
                end_times[utt_id_last+'-'+str(sub_utt_id_last)] = ent_time_last
            sub_utt = []
            start_times[utt_id+'-'+str(sub_utt_id)] = start
            is_oov_last = False
        if word == '<eps>':
            is_oov = True
            end_times[utt_id+'-'+str(sub_utt_id)] = start + dur
        elif word in vocab:
            is_oov = True
            sub_utt.append(word)
            end_times[utt_id+'-'+str(sub_utt_id)] = start + dur
        else:
            is_oov = False
            if is_oov_last == True:
                sub_utt.append(word)
                sub_utts[utt_id+'-'+str(sub_utt_id_last)] = (utt_id, sub_utt)
                end_times[utt_id+'-'+str(sub_utt_id_last)] = start + dur
                sub_utt_id += 1
            sub_utt = [word]
            start_times[utt_id+'-'+str(sub_utt_id)] = start
        utt_id_last = utt_id
        sub_utt_id_last = sub_utt_id
        is_oov_last = is_oov
        ent_time_last = start + dur
        
    if is_oov:
        if word != '<eps>':
            sub_utt.append(word)
        sub_utts[utt_id+'-'+str(sub_utt_id_last)] = (utt_id, sub_utt)
        end_times[utt_id+'-'+str(sub_utt_id_last)] = start + dur

    for utt,v in sorted(sub_utts.items()):
        print(utt, ' '.join(sub_utts[utt][1]), file=args.text_handle)
        print(utt, sub_utts[utt][0], start_times[utt], end_times[utt], file=args.subsegment_handle)

def ReadVocab(vocab_file_handle):
    vocab = set()
    if vocab_file_handle:
        for line in vocab_file_handle.readlines():
            splits = line.strip().split()
            if len(splits) == 0:
                continue
            if len(splits) > 1:
                raise Exception('Invalid format of line ' + line
                                    + ' in vocab file.')
            word = splits[0]
            vocab.add(word)
    return vocab

def Main():
    args = GetArgs()

    vocab = ReadVocab(args.vocab_handle)
    GetSubsegments(args, vocab)
   
if __name__ == "__main__":
    Main()
