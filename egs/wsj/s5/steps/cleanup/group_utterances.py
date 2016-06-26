#!/usr/bin/env python

from __future__ import print_function
import sys, argparse

def GetArgs():
    parser = argparse.ArgumentParser(description="""
    Group utterances to create common LM""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--words-threshold", type = int, default = 100,
                        help = "Combine utterances until the text has these many words")
    parser.add_argument("text_in", type = str,
                        help = "Input text file")
    parser.add_argument("text_out", type = str,
                        help = "Output grouped text file")
    parser.add_argument("orig2utt", type = str,
                        help = "Mapping from grouped utterance to the "
                        "utterance ids in the group")

    args = parser.parse_args()

    print(" ".join(sys.argv), file = sys.stderr)

    return args

def Main():
    args = GetArgs()

    text_handle = open(args.text_in)
    text_out_handle = open(args.text_out, 'w')
    orig2utt_handle = open(args.orig2utt, 'w')

    text = []
    utt_ids = []
    words_selected = 0
    for line in text_handle.readlines():
        splits = line.strip().split()
        utt_id = splits[0]
        utt_ids.append(utt_id)
        text.extend(splits[1:])

        this_num_words = len(splits) - 1

        if words_selected + this_num_words > args.words_threshold:
            extended_id = "_".join(utt_ids)
            print("{0} {1}".format(extended_id, " ".join(text)), file = text_out_handle)
            print("{0} {1}".format(extended_id, " ".join(utt_ids)), file = orig2utt_handle)
            text = []
            utt_ids = []
            words_selected = 0

    if len(utt_ids) > 0:
        extended_id = "_".join(utt_ids)
        print("{0} {1}".format(extended_id, " ".join(text)), file = text_out_handle)
        print("{0} {1}".format(extended_id, " ".join(utt_ids)), file = orig2utt_handle)

if __name__ == "__main__":
    Main()
