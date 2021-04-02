#!/usr/bin/env python3

from __future__ import print_function
import sys
import argparse
import math
import subprocess
from collections import defaultdict

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding="utf8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer,encoding="utf8")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer,encoding="utf8")

parser = argparse.ArgumentParser(description="""
This script is a wrapper for make_one_biased_lm.py that reads a Kaldi archive
of (integerized) text data from the standard input and writes a Kaldi archive of
backoff-language-model FSTs to the standard-output.  It takes care of
grouping utterances to respect the --min-words-per-graph option.  It writes
the graphs to the standard output and also outputs a map from input utterance-ids
to the per-group utterance-ids that index the output graphs.""")

parser.add_argument("--lm-opts", type = str, default = "",
                    help = "Options to pass in to make_one_biased_lm.py (which "
                    "creates the individual LM graphs), e.g. '--word-disambig-symbol=8721'.")
parser.add_argument("--min-words-per-graph", type = int, default = 100,
                    help = "Minimum number of words per utterance group; this program "
                    "will try to arrange the input utterances into groups such that each "
                    "one has at least this many words in total.")
parser.add_argument("utterance_map", type = str,
                    help = "Filename to which a map from input utterances to grouped "
                    "utterances, is written")

args = parser.parse_args()



try:
    utterance_map_file = open(args.utterance_map, "w", encoding="utf-8")
except:
    sys.exit("make_biased_lms.py: error opening {0} to write utterance map".format(
            args.utterance_map))

# This processes one group of input lines; 'group_of_lines' is
# an array of lines of input integerized text, e.g.
# [ 'utt1 67 89 432', 'utt2 89 48 62' ]
def ProcessGroupOfLines(group_of_lines):
    num_lines = len(group_of_lines)
    try:
        first_utterance_id = group_of_lines[0].split()[0]
    except:
        sys.exit("make_biased_lms.py: empty input line")

    group_utterance_id = '{0}-group-of-{1}'.format(first_utterance_id, num_lines)
    # print the group utterance-id to the stdout; it forms the name in
    # the text-form archive.
    print(group_utterance_id)
    sys.stdout.flush()

    try:
        command = "steps/cleanup/internal/make_one_biased_lm.py " + args.lm_opts
        p = subprocess.Popen(command, shell = True, stdin = subprocess.PIPE,
                             stdout = sys.stdout, stderr = sys.stderr)
        for line in group_of_lines:
            a = line.split()
            if len(a) == 0:
                sys.exit("make_biased_lms.py: empty input line")
            utterance_id = a[0]
            # print <utt> <utt-group> to utterance-map file
            print(utterance_id, group_utterance_id, file = utterance_map_file)
            rest_of_line = ' '.join(a[1:]) + '\n' # get rid of utterance id.
            p.stdin.write(rest_of_line.encode('utf-8'))
        p.stdin.close()
        assert p.wait() == 0
    except Exception:
        sys.stderr.write(
            "make_biased_lms.py: error calling subprocess, command was: " +
            command)
        raise
    # Print a blank line; this terminates the FST in the Kaldi fst-archive
    # format.
    print("")
    sys.stdout.flush()



num_words_this_group = 0
this_group_of_lines = []  # An array of strings, one per line

while True:
    line = sys.stdin.readline();
    num_words_this_group += len(line.split())
    if line != '':
        this_group_of_lines.append(line)
    if num_words_this_group >= args.min_words_per_graph or \
        (line == '' and len(this_group_of_lines) != 0):
        ProcessGroupOfLines(this_group_of_lines)
        num_words_this_group = 0
        this_group_of_lines = []
    if line == '':
        break


# test comand [to be run from ../..]
#

# (echo 1 0.5; echo 2 0.25) > top_words.txt
# (echo utt1 6 7 8 4; echo utt2 7 8 9; echo utt3 7 8) | steps/cleanup/make_biased_lms.py --lm-opts='--word-disambig-symbol=1000 --top-words=top_words.txt' foo; cat foo

# (echo utt1 6 7 8 4; echo utt2 7 8 9; echo utt3 7 8) | steps/cleanup/make_biased_lms.py --min-words-per-graph=4 --lm-opts='--word-disambig-symbol=1000 --top-words=top_words.txt' foo; cat foo
