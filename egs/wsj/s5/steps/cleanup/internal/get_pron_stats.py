#!/usr/bin/env python

# Copyright 2016  Xiaohui Zhang
# Apache 2.0.

from __future__ import print_function
import argparse
import sys
import warnings

# Collect pronounciation stats from a ctm_prons.txt file (containing entries: utt_id word phone_1...phone_N
# which must be consecutive in time) and output the stats (format: count word pronounciation) into prons.txt
def GetArgs():
    parser = argparse.ArgumentParser(description = "Accumulate pronounciation statistics from "
                                     "a ctm_prons.txt file.",
                                     epilog = "See steps/cleanup/debug_lexicon.sh for example")
    parser.add_argument("ctm_prons_file", metavar = "<ctm-prons-file>", type = str,
                        help = "File containing word-pronounciation alignments obtained from a ctm file; "
                        "each line must be <utt_id> <word> <phones>")
    parser.add_argument("silphone_file", metavar = "<silphone-file>", type = str,
                        help = "File containing a list of silence phones.")
    parser.add_argument("stats_file", metavar = "<stats-file>", type = str,
                        help = "Write accumulated statitistics to this file;"
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
    args.silphone_file_handle = open(args.silphone_file)
    if args.stats_file == "-":
        args.stats_file_handle = sys.stdout
    else:
        args.stats_file_handle = open(args.stats_file, "w")
    return args

def ReadSilPhones(silphone_file_handle):
    silphones = set()
    for line in silphone_file_handle:
        silphones.add(line.strip())
    return silphones

# Basically, this function generates an "info" list from a ctm_prons file.
# Each entry in the list represents the pronounciation candidate(s) of a word.
# For each non-<eps> word, the entry is a list: [utt_id, word, set(prnounciation_candidates)].
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

def GetStatsFromCtmProns(silphones, ctm_prons_file_handle):
    info = []
    for line in ctm_prons_file_handle.readlines():
        splits = line.strip().split()
        utt = splits[0]
        word = splits[1]
        phones_list = splits[2:]
        phones = " ".join(phones_list)
        # extract the nonsil_left and nonsil_right segments, and then try to
        # append nonsil_left to the pron candidates of preceding word, getting
        # extended pron candidates.
        if word == '<eps>':
            nonsil_left = []
            nonsil_right = [] 
            for phone in splits[2:]:
                if phone in silphones:
                    break
                nonsil_left.append(phone)
            
            for phone in reversed(splits[2:]):
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
                info[-1][2] = info[-1][2].union(pron_ext)
            if len(nonsil_right) > 0:
                info.append([utt, word, " ".join(nonsil_right)])
        else:
            prons = set()
            if phones != '':
               prons.add(phones)
            if word == 'fda':
                print(info[-1][2], phones, utt)
            # If there's a preceding <eps>, we append it's nonsil_right segment
            # to the pron candidates of the current word.
            if len(info) > 0 and utt == info[-1][0] and info[-1][1] == '<eps>' and (phones == '' or phones_list[0] not in silphones):
                # info[-1][2] is the nonsil_right segment of the phones aligned to the last <eps>.
                prons.add(info[-1][2]+' '+phones)
            info.append([utt, word, prons])
    
    stats = {}
    for utt, word, pron_set in info:
        if word != '<eps>' and len(pron_set) > 0:
            count = 1.0 / float(len(pron_set))
            for pron in pron_set:
                # remove silence phones inserted in the prons
                phones = " ".join([phone for phone in pron.split() if phone not in silphones])
                stats[(word, phones)] = stats.get((word, phones), 0) + count
    return stats

def WriteStats(stats, file_handle):            
    for word_pron, count in stats.iteritems():
        if word_pron[1] != '':
            print('{0} {1} {2}'.format(count, word_pron[0], word_pron[1]), file=file_handle)
    file_handle.close()

def Main():
    args = GetArgs()
    silphones = ReadSilPhones(args.silphone_file_handle)
    stats = GetStatsFromCtmProns(silphones, args.ctm_prons_file_handle)
    WriteStats(stats, args.stats_file_handle)            

if __name__ == "__main__":
    Main()
