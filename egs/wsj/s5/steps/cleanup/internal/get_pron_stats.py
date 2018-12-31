#!/usr/bin/env python

# Copyright 2016  Xiaohui Zhang
# Apache 2.0.

from __future__ import print_function
from __future__ import division
import argparse
import sys
import warnings

# Collect pronounciation stats from a ctm_prons.txt file of the form output
# by steps/cleanup/debug_lexicon.sh.  This input file has lines of the form:
#  utt_id word phone1 phone2 .. phoneN
#  e.g.
#  foo-bar123-342  hello h eh l l ow
# (and this script does require that lines from the same utterance be ordered in
# order of time).
# The output of this program is word pronunciation stats of the form:
#  count word phone1 .. phoneN
#  e.g.:
#  24.0  hello h ax l l ow
# This program uses various heuristics to account for the fact that in the input ctm_prons.txt
# file may not always be well aligned.  As a result of some of these heuristics the counts will
# not always be integers.

def GetArgs():
    parser = argparse.ArgumentParser(description = "Accumulate pronounciation statistics from "
                                     "a ctm_prons.txt file.",
                                     epilog = "See steps/cleanup/debug_lexicon.sh for example")
    parser.add_argument("ctm_prons_file", metavar = "<ctm-prons-file>", type = str,
                        help = "File containing word-pronounciation alignments obtained from a ctm file; "
                        "It represents phonetic decoding results, aligned with word boundaries obtained"
                        "from forced alignments."
                        "each line must be <utt_id> <word> <phones>")
    parser.add_argument("silence_file", metavar = "<silphone-file>", type = str,
                        help = "File containing a list of silence phones.")
    parser.add_argument("optional_silence_file", metavar = "<optional_silence>", type = str,
                        help = "File containing the optional silence phone. We'll be replacing empty prons by this,"
                        "because empty prons would cause a problem for lattice word alignment.")
    parser.add_argument("non_scored_words_file", metavar = "<non-scored-words-file>", type = str,
                        help = "File containing a list of non-scored words.")
    parser.add_argument("stats_file", metavar = "<stats-file>", type = str,
                        help = "Write accumulated statitistics to this file; each line represents how many times "
                        "a specific word-pronunciation pair appears in the phonetic decoding results (ctm_pron_file)."
                        "each line is <count> <word> <phones>")
    print (' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if args.ctm_prons_file == "-":
        args.ctm_prons_file_handle = sys.stdin
    else:
        args.ctm_prons_file_handle = open(args.ctm_prons_file)
    args.non_scored_words_file_handle = open(args.non_scored_words_file)
    args.silence_file_handle = open(args.silence_file)
    args.optional_silence_file_handle = open(args.optional_silence_file)
    if args.stats_file == "-":
        args.stats_file_handle = sys.stdout
    else:
        args.stats_file_handle = open(args.stats_file, "w")
    return args

def ReadEntries(file_handle):
    entries = set()
    for line in file_handle:
        entries.add(line.strip())
    return entries

# Basically, this function generates an "info" list from a ctm_prons file.
# Each entry in the list represents the pronounciation candidate(s) of a word.
# For each non-<eps> word, the entry is a list: [utt_id, word, set(pronunciation_candidates)]. e.g:
# [911Mothers_2010W-0010916-0012901-1, other, set('AH DH ER', 'AH DH ER K AH N')]
# For each <eps>, we split the phones it aligns to into two parts: "nonsil_left", 
# which includes phones before the first silphone, and "nonsil_right", which includes
# phones after the last silphone. For example, for <eps> : 'V SIL B AH SIL', 
# nonsil_left is 'V' and nonsil_right is empty ''. After processing an <eps> entry
# in ctm_prons, we put it in "info" as an entry:  [utt_id, word, nonsil_right]
# only if it's nonsil_right segment is not empty, which may be used when processing
# the next word.
# 
# Normally, one non-<eps> word is only aligned to one pronounciation candidate. However
# when there is a preceding/following <eps>, like in the following example, we
# assume the phones aligned to <eps> should be statistically distributed
# to its neighboring words (BTW we assume there are no consecutive <eps> within an utterance.)
# Thus we append the "nonsil_left" segment of these phones to the pronounciation
# of the preceding word, if the last phone of this pronounciation is not a silence phone,
# Similarly we can add a pron candidate to the following word.
# 
# For example, for the following part of a ctm_prons file:
# 911Mothers_2010W-0010916-0012901-1 other AH DH ER
# 911Mothers_2010W-0010916-0012901-1 <eps> K AH N SIL B
# 911Mothers_2010W-0010916-0012901-1 because IH K HH W AA Z AH
# 911Mothers_2010W-0010916-0012901-1 <eps> V SIL
# 911Mothers_2010W-0010916-0012901-1 when W EH N
# 911Mothers_2010W-0010916-0012901-1 people P IY P AH L
# 911Mothers_2010W-0010916-0012901-1 <eps> SIL
# 911Mothers_2010W-0010916-0012901-1 heard HH ER 
# 911Mothers_2010W-0010916-0012901-1 <eps> D
# 911Mothers_2010W-0010916-0012901-1 that SIL DH AH T
# 911Mothers_2010W-0010916-0012901-1 my M AY
# 
# The corresponding segment in the "info" list is:
# [911Mothers_2010W-0010916-0012901-1, other, set('AH DH ER', 'AH DH ER K AH N')]
# [911Mothers_2010W-0010916-0012901-1, <eps>, 'B'
# [911Mothers_2010W-0010916-0012901-1, because, set('IH K HH W AA Z AH', 'B IH K HH W AA Z AH', 'IH K HH W AA Z AH V', 'B IH K HH W AA Z AH V')]
# [911Mothers_2010W-0010916-0012901-1, when, set('W EH N')]
# [911Mothers_2010W-0010916-0012901-1, people, set('P IY P AH L')]
# [911Mothers_2010W-0010916-0012901-1, <eps>, 'D']
# [911Mothers_2010W-0010916-0012901-1, that, set('SIL DH AH T')]
# [911Mothers_2010W-0010916-0012901-1, my, set('M AY')]
# 
# Then we accumulate pronouciation stats from "info". Basically, for each occurence
# of a word, each pronounciation candidate gets equal soft counts. e.g. In the above
# example, each pron candidate of "because" gets a count of 1/4. The stats is stored
# in a dictionary (word, pron) : count.

def GetStatsFromCtmProns(silphones, optional_silence, non_scored_words, ctm_prons_file_handle):
    info = []
    for line in ctm_prons_file_handle.readlines():
        splits = line.strip().split()
        utt = splits[0]
        word = splits[1]
        phones = splits[2:]
        if phones == []:
            phones = [optional_silence]
        # extract the nonsil_left and nonsil_right segments, and then try to
        # append nonsil_left to the pron candidates of preceding word, getting
        # extended pron candidates.
        # Note: the ctm_pron file may have cases like:
        # KevinStone_2010U-0024782-0025580-1 [UH] EH
        # KevinStone_2010U-0024782-0025580-1 fda F T
        # KevinStone_2010U-0024782-0025580-1 [NOISE] IY EY
        # which means non-scored-words (except oov symbol <unk>/<UNK>) behaves like <eps>.
        # So we apply the same merging method in these cases.
        if word == '<eps>' or (word in non_scored_words and word != '<unk>' and word != '<UNK>'):
            nonsil_left = []
            nonsil_right = [] 
            for phone in phones:
                if phone in silphones:
                    break
                nonsil_left.append(phone)
            
            for phone in reversed(phones):
                if phone in silphones:
                    break
                nonsil_right.insert(0, phone)
            
            # info[-1][0] is the utt_id of the last entry
            if len(nonsil_left) > 0 and len(info) > 0 and utt == info[-1][0]: 
                # pron_ext is a set of extended pron candidates. 
                pron_ext = set()
                # info[-1][2] is the set of pron candidates of the last entry.
                for pron in info[-1][2]:
                    # skip generating the extended pron candidate if
                    # the pron ends with a silphone.
                    ends_with_sil = False
                    for sil in silphones:
                        if pron.endswith(sil):
                            ends_with_sil = True
                    if not ends_with_sil:
                        pron_ext.add(pron+" "+" ".join(nonsil_left))
                if isinstance(info[-1][2], set):
                    info[-1][2] = info[-1][2].union(pron_ext)
            if len(nonsil_right) > 0:
                info.append([utt, word, " ".join(nonsil_right)])
        else:
            prons = set()
            prons.add(" ".join(phones))
            # If there's a preceding <eps>/non_scored_words (which means the third field is a string rather than a set of strings),
            # we append it's nonsil_right segment to the pron candidates of the current word.
            if len(info) > 0 and utt == info[-1][0] and isinstance(info[-1][2], str) and (phones == [] or phones[0] not in silphones):
                # info[-1][2] is the nonsil_right segment of the phones aligned to the last <eps>/non_scored_words.
                prons.add(info[-1][2]+' '+" ".join(phones))
            info.append([utt, word, prons])
    stats = {}
    for utt, word, prons in info:
        # If the prons is not a set, the current word must be <eps> or an non_scored_word,
        # where we just left the nonsil_right part as prons.
        if isinstance(prons, set) and len(prons) > 0:
            count = 1.0 / float(len(prons))
            for pron in prons:
                phones = pron.strip().split()
                # post-processing: remove all begining/trailing silence phones.
                # we allow only candidates that either consist of a single silence
                # phone, or the silence phones are inside non-silence phones.
                if len(phones) > 1:
                    begin = 0
                    for phone in phones:
                        if phone in silphones:
                            begin += 1
                        else:
                            break
                    if begin == len(phones):
                        begin -= 1
                    phones = phones[begin:]
                    if len(phones) == 1:
                        break
                    end = len(phones)
                    for phone in reversed(phones):
                        if phone in silphones:
                            end -= 1
                        else:
                            break
                    phones = phones[:end]
                phones = " ".join(phones)
                stats[(word, phones)] = stats.get((word, phones), 0) + count
    return stats

def WriteStats(stats, file_handle):            
    for word_pron, count in stats.items():
        print('{0} {1} {2}'.format(count, word_pron[0], word_pron[1]), file=file_handle)
    file_handle.close()

def Main():
    args = GetArgs()
    silphones = ReadEntries(args.silence_file_handle)
    non_scored_words = ReadEntries(args.non_scored_words_file_handle)
    optional_silence = ReadEntries(args.optional_silence_file_handle)
    stats = GetStatsFromCtmProns(silphones, optional_silence.pop(), non_scored_words, args.ctm_prons_file_handle)
    WriteStats(stats, args.stats_file_handle)            

if __name__ == "__main__":
    Main()
