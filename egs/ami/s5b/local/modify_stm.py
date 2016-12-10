#! /usr/bin/env python

import sys
import collections
import itertools
import argparse

from collections import defaultdict

def IgnoreWordList(stm_lines, wordlist):
    for i in range(0, len(stm_lines)):
        line = stm_lines[i]
        splits = line.strip().split()

        line_changed = False
        for j in range(5, len(splits)):
            if str.lower(splits[j]) in wordlist:
                splits[j] = "{{ {0} / @ }}".format(splits[j])
                line_changed = True


        if line_changed:
            stm_lines[i] = " ".join(splits)

def IgnoreIsolatedWords(stm_lines):
    for i in range(0, len(stm_lines)):
        line = stm_lines[i]
        splits = line.strip().split()

        assert( splits[5][0] != '<' )

        if len(splits) == 6 and splits[5] != "IGNORE_TIME_SEGMENT_IN_SCORING":
            splits.insert(5, "<ISO>")
        else:
            splits.insert(5, "<NO_ISO>")
        stm_lines[i] = " ".join(splits)

def IgnoreBeginnings(stm_lines):
    beg_times = defaultdict(itertools.repeat(float("inf")).next)

    lines_to_add = []
    for line in stm_lines:
        splits = line.strip().split()

        beg_times[(splits[0],splits[1])] = min(beg_times[(splits[0],splits[1])], float(splits[3]))

    for t,v in beg_times.iteritems():
        lines_to_add.append("{0} {1} {0} 0.0 {2} <NO_ISO> IGNORE_TIME_SEGMENT_IN_SCORING".format(t[0], t[1], v))

    stm_lines.extend(lines_to_add)

def WriteStmLines(stm_lines):
    for line in stm_lines:
        print(line)

def GetArgs():
    parser = argparse.ArgumentParser("This script modifies STM to remove certain words and segments from scoring. Use sort +0 -1 +1 -2 +3nb -4 while writing out.",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--ignore-beginnings",
                        type = str, choices = ["true", "false"],
                        help = "Ignore beginnings of the recordings since "
                        "they are not transcribed")
    parser.add_argument("--ignore-isolated-words",
                        type = str, choices = ["true", "false"],
                        help = "Remove isolated words from scoring "
                        "because they may be hard to recognize without "
                        "speaker diarization")
    parser.add_argument("--ignore-word-list",
                        type = str,
                        help = "List of words to be ignored")

    args = parser.parse_args()

    return args

def Main():
    args = GetArgs()

    stm_lines = [ x.strip() for x in sys.stdin.readlines() ]

    print (';; LABEL "NO_ISO", "No isolated words", "Ignoring isolated words"')
    print (';; LABEL "ISO", "Isolated words", "isolated words"')

    #if args.ignore_word_list is not None:
    #    wordlist = {}
    #    for x in open(args.ignore_word_list).readlines():
    #        wordlist[str.lower(x.strip())] = 1
    #    IgnoreWordList(stm_lines, wordlist)

    IgnoreIsolatedWords(stm_lines)
    IgnoreBeginnings(stm_lines)

    WriteStmLines(stm_lines)

if __name__ == "__main__":
    Main()
