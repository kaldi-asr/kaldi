#!/usr/bin/env python

# Copyright 2016   Vimal Manohar
#           2016   Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

from __future__ import print_function
import sys, operator, argparse, os
from collections import defaultdict

# This script reads and writes the 'ctm-edits' file that is
# produced by get_ctm_edits.py.
#
# It is to be applied after modify_ctm_edits.py.  Its function is to add, in
# certain circumstances, an optional extra field with the word 'tainted' to the
# ctm-edits format, e.g an input line like:
#
# AJJacobs_2007P-0001605-0003029 1 0 0.09 <eps> 1.0 <eps> sil
# might become:
# AJJacobs_2007P-0001605-0003029 1 0 0.09 <eps> 1.0 <eps> sil tainted
#
# It also deletes certain lines, representing deletions, from the ctm (if they
# were next to taintable lines... their presence could then be inferred from the
# 'tainted' flag).
#
# You should interpret the 'tainted' flag as "we're not sure what's going on here;
# don't trust this."
#
# One of the problem this script is trying to solve is that if we have errors
# that are adjacent to silence or non-scored words
# it's not at all clear whether the silence or non-scored words were really such,
# or might have contained actual words.
# Also, if we have words in the reference that were realized as '<unk>' in the
# hypothesis, and they are adjacent to errors, it's almost always the case
# that the '<unk>' doesn't really correspond to the word in the reference, so
# we mark these as 'tainted'.
#
# The rule for tainting is quite simple; see the code.



parser = argparse.ArgumentParser(
    description = "This program modifies the ctm-edits format to identify "
    "silence and 'fixed' non-scored-word lines, and lines where the hyp is "
    "<unk> and the reference is a real but OOV word, where there is a relatively "
    "high probability that something is going wrong so we shouldn't trust "
    "this line.  It adds the field 'tainted' to such "
    "lines.  Lines in the ctm representing deletions from the reference will "
    "be removed if they have 'tainted' adjacent lines (since it won't be clear "
    "where such reference words were really realized, if at all). "
    "See comments at the top of the script for more information.")

parser.add_argument("--verbose", type = int, default = 1,
                    choices=[0,1,2,3],
                    help = "Verbose level, higher = more verbose output")
parser.add_argument("--remove-deletions", type=str, default="true",
                    choices=["true", "false"],
                    help = "Remove deletions next to taintable lines")
parser.add_argument("ctm_edits_in", metavar = "<ctm-edits-in>",
                    help = "Filename of input ctm-edits file. "
                    "Use /dev/stdin for standard input.")
parser.add_argument("ctm_edits_out", metavar = "<ctm-edits-out>",
                    help = "Filename of output ctm-edits file. "
                    "Use /dev/stdout for standard output.")

args = parser.parse_args()
args.remove_deletions = bool(args.remove_deletions == "true")



# This function is the core of the program, that does the tainting and
# removes some lines representing deletions.
# split_lines_of_utt is a list of lists, one per line, each containing the
# sequence of fields.  Returns the same format of data after processing to add
# the 'tainted' field.  Note: this function is destructive of its input; the
# input will not have the same value afterwards.
def ProcessUtterance(split_lines_of_utt, remove_deletions=True):
    global num_lines_of_type, num_tainted_lines, \
           num_del_lines_giving_taint, num_sub_lines_giving_taint, \
           num_ins_lines_giving_taint

    # work out whether each line is taintable [i.e. silence or fix or unk replacing
    # real-word].
    taintable = [ False ] * len(split_lines_of_utt)
    for i in range(len(split_lines_of_utt)):
        edit_type = split_lines_of_utt[i][7]
        if edit_type == 'sil' or edit_type == 'fix':
            taintable[i] = True
        elif edit_type == 'cor' and split_lines_of_utt[i][4] != split_lines_of_utt[i][6]:
            # this is the case when <unk> replaces a real word that was out of
            # the vocabulary; we mark it as correct because such words do
            # translate to <unk> if we don't have a pronunciations.  However we
            # don't have good confidence that the alignments of such words are
            # accurate if they are adjacent to errors.
            taintable[i] = True


    for i in range(len(split_lines_of_utt)):
        edit_type = split_lines_of_utt[i][7]
        num_lines_of_type[edit_type] += 1
        if edit_type == 'del' or edit_type == 'sub' or edit_type == 'ins':
            tainted_an_adjacent_line = False
            # First go backwards tainting lines
            j = i - 1
            while j >= 0 and taintable[j]:
                tainted_an_adjacent_line = True
                if len(split_lines_of_utt[j]) == 8:
                    num_tainted_lines += 1
                    split_lines_of_utt[j].append('tainted')
                j -= 1
            # Next go forwards tainting lines
            j = i + 1
            while j < len(split_lines_of_utt) and taintable[j]:
                tainted_an_adjacent_line = True
                if len(split_lines_of_utt[j]) == 8:
                    num_tainted_lines += 1
                    split_lines_of_utt[j].append('tainted')
                j += 1
            if tainted_an_adjacent_line:
                if edit_type == 'del':
                    if remove_deletions:
                        split_lines_of_utt[i][7] = 'remove-this-line'
                    num_del_lines_giving_taint += 1
                elif edit_type == 'sub':
                    num_sub_lines_giving_taint += 1
                else:
                    num_ins_lines_giving_taint += 1

    new_split_lines_of_utt = []
    for i in range(len(split_lines_of_utt)):
        if (not remove_deletions
                or split_lines_of_utt[i][7] != 'remove-this-line'):
            new_split_lines_of_utt.append(split_lines_of_utt[i])
    return new_split_lines_of_utt


def ProcessData():
    try:
        f_in = open(args.ctm_edits_in)
    except:
        sys.exit("taint_ctm_edits.py: error opening ctm-edits input "
                 "file {0}".format(args.ctm_edits_in))
    try:
        f_out = open(args.ctm_edits_out, 'w')
    except:
        sys.exit("taint_ctm_edits.py: error opening ctm-edits output "
                 "file {0}".format(args.ctm_edits_out))
    num_lines_processed = 0


    # Most of what we're doing in the lines below is splitting the input lines
    # and grouping them per utterance, before giving them to ProcessUtterance()
    # and then printing the modified lines.
    first_line = f_in.readline()
    if first_line == '':
        sys.exit("taint_ctm_edits.py: empty input")
    split_pending_line = first_line.split()
    if len(split_pending_line) == 0:
        sys.exit("taint_ctm_edits.py: bad input line " + first_line)
    cur_utterance = split_pending_line[0]
    split_lines_of_cur_utterance = []

    while True:
        if len(split_pending_line) == 0 or split_pending_line[0] != cur_utterance:
            split_lines_of_cur_utterance = ProcessUtterance(
                split_lines_of_cur_utterance, args.remove_deletions)
            for split_line in split_lines_of_cur_utterance:
                print(' '.join(split_line), file = f_out)
            split_lines_of_cur_utterance = []
            if len(split_pending_line) == 0:
                break
            else:
                cur_utterance = split_pending_line[0]

        split_lines_of_cur_utterance.append(split_pending_line)
        next_line = f_in.readline()
        split_pending_line = next_line.split()
        if len(split_pending_line) == 0:
            if next_line != '':
                sys.exit("taint_ctm_edits.py: got an empty or whitespace input line")
    try:
        f_out.close()
    except:
        sys.exit("taint_ctm_edits.py: error closing ctm-edits output "
                 "(broken pipe or full disk?)")

def PrintNonScoredStats():
    if args.verbose < 1:
        return
    if num_lines == 0:
        print("taint_ctm_edits.py: processed no input.", file = sys.stderr)
    num_lines_modified = sum(ref_change_stats.values())
    num_incorrect_lines = num_lines - num_correct_lines
    percent_lines_incorrect= '%.2f' % (num_incorrect_lines * 100.0 / num_lines)
    percent_modified = '%.2f' % (num_lines_modified * 100.0 / num_lines);
    percent_of_incorrect_modified = '%.2f' % (num_lines_modified * 100.0 / num_incorrect_lines)
    print("taint_ctm_edits.py: processed {0} lines of ctm ({1}% of which incorrect), "
          "of which {2} were changed fixing the reference for non-scored words "
          "({3}% of lines, or {4}% of incorrect lines)".format(
            num_lines, percent_lines_incorrect, num_lines_modified,
            percent_modified, percent_of_incorrect_modified),
          file = sys.stderr)

    keys = sorted(list(ref_change_stats.keys()), reverse=True,
                  key = lambda x: ref_change_stats[x])
    num_keys_to_print = 40 if args.verbose >= 2 else 10

    print("taint_ctm_edits.py: most common edits (as percentages "
          "of all such edits) are:\n" +
          ('\n'.join([ '%s [%.2f%%]' % (k, ref_change_stats[k]*100.0/num_lines_modified)
                     for k in keys[0:num_keys_to_print]]))
          + '\n...'if num_keys_to_print < len(keys) else '',
          file = sys.stderr)


def PrintStats():
    tot_lines = sum(num_lines_of_type.values())
    if args.verbose < 1 or tot_lines == 0:
        return
    print("taint_ctm_edits.py: processed {0} input lines, whose edit-types were: ".format(tot_lines) +
          ', '.join([ '%s = %.2f%%' % (k, num_lines_of_type[k] * 100.0 / tot_lines)
                      for k in sorted(list(num_lines_of_type.keys()), reverse = True,
                                      key = lambda k: num_lines_of_type[k])  ]),
          file = sys.stderr)


    del_giving_taint_percent = num_del_lines_giving_taint * 100.0 / tot_lines
    sub_giving_taint_percent = num_sub_lines_giving_taint * 100.0 / tot_lines
    ins_giving_taint_percent = num_ins_lines_giving_taint * 100.0 / tot_lines
    tainted_lines_percent = num_tainted_lines * 100.0 / tot_lines

    print("taint_ctm_edits.py: as a percentage of all lines, (%.2f%%, %.2f%%, %.2f%%) were "
          "(deletions, substitutions, insertions) that tainted adjacent lines.  %.2f%% of all "
          "lines were tainted." % (del_giving_taint_percent, sub_giving_taint_percent,
                                   ins_giving_taint_percent, tainted_lines_percent),
          file = sys.stderr)



# num_lines_of_type will map from line-type ('cor', 'sub', etc.) to count.
num_lines_of_type = defaultdict(int)
num_tainted_lines = 0
num_del_lines_giving_taint = 0
num_sub_lines_giving_taint = 0
num_ins_lines_giving_taint = 0

ProcessData()
PrintStats()

