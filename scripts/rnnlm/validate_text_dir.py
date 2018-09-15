#!/usr/bin/env python3

# Copyright  2017  Jian Wang
# License: Apache 2.0.

import os
import argparse
import sys

import re
tab_or_space = re.compile('[ \t]+')

parser = argparse.ArgumentParser(description="Validates data directory containing text "
                                 "files from one or more data sources, including dev.txt.",
                                 epilog="E.g. " + sys.argv[0] + " data/rnnlm/data",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--spot-check", type=str, default='true',
                    choices=['true', 'false'],
                    help="If true, only do spot check on text files.")
parser.add_argument("--allow-internal-eos", type=str, default='true',
                    choices=['true', 'false'],
                    help="If true, allow internal </s> in lines of the text.")
parser.add_argument("text_dir",
                    help="Directory in which to look for text data")

args = parser.parse_args()

EOS_SYMBOL = '</s>'
SPECIAL_SYMBOLS = ['<s>', '<brk>', '<eps>']

if not os.path.exists(args.text_dir):
    sys.exit(sys.argv[0] + ": Expected directory {0} to exist".format(args.text_dir))

if not os.path.exists("{0}/dev.txt".format(args.text_dir)):
    sys.exit(sys.argv[0] + ": Expected file {0}/dev.txt to exist".format(args.text_dir))


num_text_files = 0


def check_text_file(text_file):
    with open(text_file, 'r', encoding="latin-1") as f:
        found_nonempty_line = False
        lineno = 0
        if args.allow_internal_eos == 'true':
            disallowed_symbols = SPECIAL_SYMBOLS
        else:
            disallowed_symbols = SPECIAL_SYMBOLS + [EOS_SYMBOL]
        for line in f:
            line = line.strip("\n")
            if line is None:
                break
            lineno += 1
            if args.spot_check == 'true' and lineno > 10:
                break
            words = re.split(tab_or_space, line)
            if len(words) != 0:
                found_nonempty_line = True
                for word in words:
                    if word in disallowed_symbols:
                        sys.exit(sys.argv[0] + ": Found suspicious line '{0}' in file {1} at {2} ({3} "
                                 " symbol is disallowed!)".format(line, text_file, lineno, word))
                if words[-1] == EOS_SYMBOL:
                    sys.exit(sys.argv[0] + ": Found suspicious line '{0}' in file {1} at {2} (EOS symbol "
                             "at the end of a line is disallowed!)".format(line, text_file, lineno))
                if len(words) >= 1000:
                    print(sys.argv[0] + ": Too long line with {0} words in file "
                          "{1} at {2}".format(len(words), text_file, lineno), file=sys.stderr)
    if not found_nonempty_line:
        sys.exit(sys.argv[0] + ": Input file {0} doesn't look right.".format(text_file))

    # Next we're going to check that it's not the case
    # that the first and second fields have disjoint words on them, and the
    # first field is always unique, which would be the case if the lines started
    # with some kind of utterance-id
    first_field_set = set()
    other_fields_set = set()
    with open(text_file, 'r', encoding="latin-1") as f:
        for line in f:
            array = re.split(tab_or_space, line)
            if len(array) > 0:
                first_word = array[0]
                if first_word in first_field_set or first_word in other_fields_set:
                    # the first field isn't always unique, or is shared with other
                    # fields.
                    return
                first_field_set.add(first_word)
            for i in range(1, len(array)):
                other_word = array[i]
                if other_word in first_field_set:
                    # the first field has a value shared by some word not in the
                    # first position.
                    return
                other_fields_set.add(other_word)
    print(sys.argv[0] + ": input file {0} looks suspicious; check that you "
          "don't have utterance-ids in the first field (i.e. you shouldn't provide "
          "lines that look like 'utterance-id1 hello there').  Ignore this warning "
          "if you don't have that problem.".format(text_file), file=sys.stderr)


for f in os.listdir(args.text_dir):
    full_path = args.text_dir + "/" + f
    if os.path.isdir(full_path) or f.endswith(".counts"):
        continue
    if not f.endswith(".txt"):
        sys.exit(sys.argv[0] + ": Text directory should not contain files with suffixes "
                 "other than .txt and .counts: " + f)
    if not os.path.isfile(full_path):
        sys.exit(sys.argv[0] + ": Expected {0} to be a file.".format(full_path))
    check_text_file(full_path)
    num_text_files += 1

if num_text_files < 2:
    sys.exit(sys.argv[0] + ": Directory {0} should contain at least one .txt file "
             "other than dev.txt.".format(args.text_dir))
