#!/usr/bin/env python

# Copyright 2016   Vimal Manohar
#           2016   Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

from __future__ import print_function
import sys, operator, argparse, os
from collections import defaultdict

# This script reads and writes the 'ctm-edits' file that is
# produced by get_ctm_edits.py.

# It modifies the ctm-edits so that non-scored words
# are not counted as errors: for instance, if there are things like
# [COUGH] and [NOISE] in the transcript, deletions, insertions and
# substitutions involving them are allowed, and we modify the reference
# to correspond to the hypothesis.
#
# If you supply the <lang> directory (the one that corresponds to
# how you decoded the data) to this script, it assumes that the <lang>
# directory contains phones/align_lexicon.int, and it uses this to work
# out a reasonable guess of the non-scored phones, based on which have
# a single-word pronunciation that maps to a silence phone.
# It then uses the words.txt to work out the written form of those words.
#
# Alternatively, you may specify a file containing the non-scored words one
# per line, with the --non-scored-words option.
#
# Non-scored words that were deleted (i.e. they were in the ref but not the
# hyp) are simply removed from the ctm.  For non-scored words that
# were inserted or substituted, we change the reference word to match the
# hyp word, but instead of marking the operation as 'cor' (correct), we
# mark it as 'fix' (fixed), so that it will not be positively counted as a correct
# word for purposes of finding the optimal segment boundaries.
#
# e.g.
# <file-id> <channel> <start-time> <duration> <conf> <hyp-word> <ref-word> <edit-type>
# [note: the <channel> will always be 1].

# AJJacobs_2007P-0001605-0003029 1 0 0.09 <eps> 1.0 <eps> sil
# AJJacobs_2007P-0001605-0003029 1 0.09 0.15 i 1.0 i cor
# AJJacobs_2007P-0001605-0003029 1 0.24 0.25 thought 1.0 thought cor
# AJJacobs_2007P-0001605-0003029 1 0.49 0.14 i'd 1.0 i'd cor
# AJJacobs_2007P-0001605-0003029 1 0.63 0.22 tell 1.0 tell cor
# AJJacobs_2007P-0001605-0003029 1 0.85 0.11 you 1.0 you cor
# AJJacobs_2007P-0001605-0003029 1 0.96 0.05 a 1.0 a cor
# AJJacobs_2007P-0001605-0003029 1 1.01 0.24 little 1.0 little cor
# AJJacobs_2007P-0001605-0003029 1 1.25 0.5 about 1.0 about cor
# AJJacobs_2007P-0001605-0003029 1 1.75 0.48 [UH] 1.0 [UH] cor


parser = argparse.ArgumentParser(
    description = "This program modifies the reference in the ctm-edits which "
    "is output by steps/cleanup/internal/get_ctm_edits.py, to allow insertions, deletions and "
    "substitutions of non-scored words, and [if --allow-repetitions=true], "
    "duplications of single words or pairs of scored words (to account for dysfluencies "
    "that were not transcribed).  Note: deletions and substitutions of non-scored words "
    "after the reference is corrected, will be marked as operation 'fix' rather than "
    "'cor' (correct) so that the downstream processing knows that this was not in "
    "the original reference.  Also by defaults tags non-scored words as such when "
    "they are correct; see the --tag-non-scored option.")

parser.add_argument("--verbose", type = int, default = 1,
                    choices=[0,1,2,3],
                    help = "Verbose level, higher = more verbose output")
parser.add_argument("--allow-repetitions", type = str, default = 'true',
                    choices=['true','false'],
                    help = "If true, allow repetitions in the transcript of one or "
                    "two-word sequences: for instance if the ref says 'i' but "
                    "the hyp says 'i i', or the ref says 'but then' and the hyp says "
                    "'but then but then', fix the reference accordingly.  Intervening "
                    "non-scored words are allowed between the repetitions.  These "
                    "fixes will be marked as 'cor', not as 'fix', since there is "
                    "generally no way to tell which repetition was the 'real' one "
                    "(and since we're generally confident that such things were "
                    "actually uttered).")
parser.add_argument("non_scored_words_in", metavar = "<non-scored-words-file>",
                    help="Filename of file containing a list of non-scored words, "
                    "one per line. See steps/cleanup/get_nonscored_words.py.")
parser.add_argument("ctm_edits_in", metavar = "<ctm-edits-in>",
                    help = "Filename of input ctm-edits file. "
                    "Use /dev/stdin for standard input.")
parser.add_argument("ctm_edits_out", metavar = "<ctm-edits-out>",
                    help = "Filename of output ctm-edits file. "
                    "Use /dev/stdout for standard output.")

args = parser.parse_args()



def ReadNonScoredWords(non_scored_words_file):
    global non_scored_words
    try:
        f = open(non_scored_words_file)
    except:
        sys.exit("modify_ctm_edits.py: error opening file: "
                 "--non-scored-words=" + non_scored_words_file)
    for line in f.readlines():
        a = line.split()
        if not len(line.split()) == 1:
            sys.exit("modify_ctm_edits.py: bad line in non-scored-words "
                     "file {0}: {1}".format(non_scored_words_file, line))
        non_scored_words.add(a[0])
    f.close()



# The ctm-edits file format is as follows [note: file-id is really utterance-id
# in this context].
# <file-id> <channel> <start-time> <duration> <conf> <hyp-word> <ref-word> <edit>
# e.g.:
# AJJacobs_2007P-0001605-0003029 1 0 0.09 <eps> 1.0 <eps> sil
# AJJacobs_2007P-0001605-0003029 1 0.09 0.15 i 1.0 i cor
# ...
# This function processes a single line of ctm-edits input for fixing
# "non-scored" words.  The input 'a' is the split line as an array of fields.
# It modifies the object 'a'.   This function returns the modified array,
# and please note that it is destructive of its input 'a'.
# If it returnso the empty array then the line is to be deleted.
def ProcessLineForNonScoredWords(a):
    global num_lines, num_correct_lines, ref_change_stats
    try:
        assert len(a) == 8
        num_lines += 1
        # we could do:
        # [ file, channel, start, duration, hyp_word, confidence, ref_word, edit_type ] = a
        duration = a[3]
        hyp_word = a[4]
        ref_word = a[6]
        edit_type = a[7]
        if edit_type == 'ins':
            assert ref_word == '<eps>'
            if hyp_word in non_scored_words:
                # insert this non-scored word into the reference.
                ref_change_stats[ref_word + ' -> ' + hyp_word] += 1
                ref_word = hyp_word
                edit_type = 'fix'
        elif edit_type == 'del':
            assert hyp_word == '<eps>' and float(duration) == 0.0
            if ref_word in non_scored_words:
                ref_change_stats[ref_word + ' -> ' + hyp_word] += 1
                return []
        elif edit_type == 'sub':
            if hyp_word in non_scored_words and ref_word in non_scored_words:
                # we also allow replacing one non-scored word with another.
                ref_change_stats[ref_word + ' -> ' + hyp_word] += 1
                ref_word = hyp_word
                edit_type = 'fix'
        else:
            assert edit_type == 'cor' or edit_type == 'sil'
            num_correct_lines += 1

        a[4] = hyp_word
        a[6] = ref_word
        a[7] = edit_type
        return a

    except Exception as e:
        print("modify_ctm_edits.py: bad line in ctm-edits input: " + ' '.join(a),
              file = sys.stderr)
        print("modify_ctm_edits.py: exception was: " + str(e),
              file = sys.stderr)
        sys.exit(1)

# This function processes the split lines of one utterance (as a
# list of lists of fields), to allow repetitions of words, so if the
# reference says 'i' but the hyp says 'i i', or the ref says
# 'you know' and the hyp says 'you know you know', we change the
# ref to match.
# It returns the modified list-of-lists [but note that the input
# is actually modified].
def ProcessUtteranceForRepetitions(split_lines_of_utt):
    global non_scored_words, repetition_stats
    # The array 'selected_lines' will contain the indexes of of selected
    # elements of 'split_lines_of_utt'.  Consider split_line =
    # split_lines_of_utt[i].  If the hyp and ref words in split_line are both
    # either '<eps>' or non-scoreable words, we discard the index.
    # Otherwise we put it into selected_lines.
    selected_line_indexes = []
    # selected_edits will contain, for each element of selected_line_indexes, the
    # corresponding edit_type from the original utterance previous to
    # this function call ('cor', 'ins', etc.).
    #
    # As a special case, if there was a substitution ('sub') where the
    # reference word was a non-scored word and the hyp word was a real word,
    # we mark it in this array as 'ins', because for purposes of this algorithm
    # it behaves the same as an insertion.
    #
    # Whenever we do any operation that will change the reference, we change
    # all the selected_edits in the array to None so that they won't match
    # any further operations.
    selected_edits = []
    # selected_hyp_words will contain, for each element of selected_line_indexes, the
    # corresponding hyp_word.
    selected_hyp_words = []

    for i in range(len(split_lines_of_utt)):
        split_line = split_lines_of_utt[i]
        hyp_word = split_line[4]
        ref_word = split_line[6]
        # keep_this_line will be True if we are going to keep this line in the
        # 'selected lines' for further processing of repetitions.  We only
        # eliminate lines involving non-scored words or epsilon in both hyp
        # and reference position
        # [note: epsilon in hyp position for non-empty segments indicates
        #  optional-silence, and it does make sense to make this 'invisible',
        #  just like non-scored words, for the purposes of this code.]
        keep_this_line = True
        if (hyp_word == '<eps>' or hyp_word in non_scored_words) and \
           (ref_word == '<eps>' or ref_word in non_scored_words):
            keep_this_line = False
        if keep_this_line:
            selected_line_indexes.append(i)
            edit_type = split_line[7]
            if edit_type == 'sub' and ref_word in non_scored_words:
                assert not hyp_word in non_scored_words
                # For purposes of this algorithm, substitution of, say,
                # '[COUGH]' by 'hello' behaves like an insertion of 'hello',
                # since we're willing to remove the '[COUGH]' from the
                # transript.
                edit_type = 'ins'
            selected_edits.append(edit_type)
            selected_hyp_words.append(hyp_word)

    # indexes_to_fix will be a list of indexes into 'selected_indexes' where we
    # plan to fix the ref to match the hyp.
    indexes_to_fix = []

    # This loop scans for, and fixes, two-word insertions that follow,
    # or precede, the corresponding correct words.
    for i in range(0, len(selected_line_indexes) - 3):
        this_indexes = selected_line_indexes[i:i+4]
        this_hyp_words = selected_hyp_words[i:i+4]

        if this_hyp_words[0] == this_hyp_words[2] and \
           this_hyp_words[1] == this_hyp_words[3] and \
           this_hyp_words[0] != this_hyp_words[1]:
            # if the hyp words were of the form [ 'a', 'b', 'a', 'b' ]...
            this_edits = selected_edits[i:i+4]
            if this_edits == [ 'cor', 'cor', 'ins', 'ins' ] or \
                    this_edits == [ 'ins', 'ins', 'cor', 'cor' ]:
                if this_edits[0] == 'cor':
                    indexes_to_fix += [ i+2, i+3 ]
                else:
                    indexes_to_fix += [ i, i+1 ]

                # the next line prevents this region of the text being used
                # in any further edits.
                selected_edits[i:i+4] = [ None, None, None, None ]
                word_pair = this_hyp_words[0] + ' '  + this_hyp_words[1]
                # e.g. word_pair = 'hi there'
                # add 2 because these stats are of words.
                repetition_stats[word_pair] += 2
                # the next line prevents this region of the text being used
                # in any further edits.
                selected_edits[i:i+4] = [ None, None, None, None ]

    # This loop scans for, and fixes, one-word insertions that follow,
    # or precede, the corresponding correct words.
    for i in range(0, len(selected_line_indexes) - 1):
        this_indexes = selected_line_indexes[i:i+2]
        this_hyp_words = selected_hyp_words[i:i+2]

        if this_hyp_words[0] == this_hyp_words[1]:
            # if the hyp words were of the form [ 'a', 'a' ]...
            this_edits = selected_edits[i:i+2]
            if this_edits == [ 'cor', 'ins' ] or this_edits == [ 'ins', 'cor' ]:
                if this_edits[0] == 'cor':
                    indexes_to_fix.append(i+1)
                else:
                    indexes_to_fix.append(i)
                repetition_stats[this_hyp_words[0]] += 1
                # the next line prevents this region of the text being used
                # in any further edits.
                selected_edits[i:i+2] = [ None, None ]

    for i in indexes_to_fix:
        j = selected_line_indexes[i]
        split_line = split_lines_of_utt[j]
        ref_word = split_line[6]
        hyp_word = split_line[4]
        assert ref_word == '<eps>' or ref_word in non_scored_words
        # we replace reference with the decoded word, which will be a
        # repetition.
        split_line[6] = hyp_word
        split_line[7] = 'cor'

    return split_lines_of_utt


# note: split_lines_of_utt is a list of lists, one per line, each containing the
# sequence of fields.
# Returns the same format of data after processing.
def ProcessUtterance(split_lines_of_utt):
    new_split_lines_of_utt = []
    for split_line in split_lines_of_utt:
        new_split_line = ProcessLineForNonScoredWords(split_line)
        if new_split_line != []:
            new_split_lines_of_utt.append(new_split_line)
    if args.allow_repetitions == 'true':
        new_split_lines_of_utt = ProcessUtteranceForRepetitions(new_split_lines_of_utt)
    return new_split_lines_of_utt


def ProcessData():
    try:
        f_in = open(args.ctm_edits_in)
    except:
        sys.exit("modify_ctm_edits.py: error opening ctm-edits input "
                 "file {0}".format(args.ctm_edits_in))
    try:
        f_out = open(args.ctm_edits_out, 'w')
    except:
        sys.exit("modify_ctm_edits.py: error opening ctm-edits output "
                 "file {0}".format(args.ctm_edits_out))
    num_lines_processed = 0


    # Most of what we're doing in the lines below is splitting the input lines
    # and grouping them per utterance, before giving them to ProcessUtterance()
    # and then printing the modified lines.
    first_line = f_in.readline()
    if first_line == '':
        sys.exit("modify_ctm_edits.py: empty input")
    split_pending_line = first_line.split()
    if len(split_pending_line) == 0:
        sys.exit("modify_ctm_edits.py: bad input line " + first_line)
    cur_utterance = split_pending_line[0]
    split_lines_of_cur_utterance = []

    while True:
        if len(split_pending_line) == 0 or split_pending_line[0] != cur_utterance:
            split_lines_of_cur_utterance = ProcessUtterance(split_lines_of_cur_utterance)
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
                sys.exit("modify_ctm_edits.py: got an empty or whitespace input line")
    try:
        f_out.close()
    except:
        sys.exit("modify_ctm_edits.py: error closing ctm-edits output "
                 "(broken pipe or full disk?)")

def PrintNonScoredStats():
    if args.verbose < 1:
        return
    if num_lines == 0:
        print("modify_ctm_edits.py: processed no input.", file = sys.stderr)
    num_lines_modified = sum(ref_change_stats.values())
    num_incorrect_lines = num_lines - num_correct_lines
    percent_lines_incorrect= '%.2f' % (num_incorrect_lines * 100.0 / num_lines)
    percent_modified = '%.2f' % (num_lines_modified * 100.0 / num_lines);
    percent_of_incorrect_modified = '%.2f' % (num_lines_modified * 100.0 / num_incorrect_lines)
    print("modify_ctm_edits.py: processed {0} lines of ctm ({1}% of which incorrect), "
          "of which {2} were changed fixing the reference for non-scored words "
          "({3}% of lines, or {4}% of incorrect lines)".format(
            num_lines, percent_lines_incorrect, num_lines_modified,
            percent_modified, percent_of_incorrect_modified),
          file = sys.stderr)

    keys = sorted(ref_change_stats.keys(), reverse=True,
                  key = lambda x: ref_change_stats[x])
    num_keys_to_print = 40 if args.verbose >= 2 else 10

    print("modify_ctm_edits.py: most common edits (as percentages "
          "of all such edits) are:\n" +
          ('\n'.join([ '%s [%.2f%%]' % (k, ref_change_stats[k]*100.0/num_lines_modified)
                     for k in keys[0:num_keys_to_print]]))
          + '\n...'if num_keys_to_print < len(keys) else '',
          file = sys.stderr)


def PrintRepetitionStats():
    if args.verbose < 1 or sum(repetition_stats.values()) == 0:
        return
    num_lines_modified = sum(repetition_stats.values())
    num_incorrect_lines = num_lines - num_correct_lines
    percent_lines_incorrect= '%.2f' % (num_incorrect_lines * 100.0 / num_lines)
    percent_modified = '%.2f' % (num_lines_modified * 100.0 / num_lines);
    percent_of_incorrect_modified = '%.2f' % (num_lines_modified * 100.0 / num_incorrect_lines)
    print("modify_ctm_edits.py: processed {0} lines of ctm ({1}% of which incorrect), "
          "of which {2} were changed fixing the reference for repetitions ({3}% of "
          "lines, or {4}% of incorrect lines)".format(
            num_lines, percent_lines_incorrect, num_lines_modified,
            percent_modified, percent_of_incorrect_modified),
          file = sys.stderr)

    keys = sorted(repetition_stats.keys(), reverse=True,
                  key = lambda x: repetition_stats[x])
    num_keys_to_print = 40 if args.verbose >= 2 else 10

    print("modify_ctm_edits.py: most common repetitions inserted into reference (as percentages "
          "of all words fixed in this way) are:\n" +
          ('\n'.join([ '%s [%.2f%%]' % (k, repetition_stats[k]*100.0/num_lines_modified)
                     for k in keys[0:num_keys_to_print]]))
          + '\n...' if num_keys_to_print < len(keys) else '',
          file = sys.stderr)


non_scored_words = set()
ReadNonScoredWords(args.non_scored_words_in)

num_lines = 0
num_correct_lines = 0
# ref_change_stats will be a map from a string like
# 'foo -> bar' to an integer count; it keeps track of how much we changed
# the reference.
ref_change_stats = defaultdict(int)
# repetition_stats will be a map from strings like
# 'a', or 'a b' (the repeated strings), to an integer count; like
# ref_change_stats, it keeps track of how many changes we made
# in allowing repetitions.
repetition_stats = defaultdict(int)

ProcessData()
PrintNonScoredStats()
PrintRepetitionStats()
