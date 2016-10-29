#!/usr/bin/env python

# Copyright 2016   Vimal Manohar
#           2016   Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

from __future__ import print_function
import sys, operator, argparse

# Modify the CTM to include for each token the information from Levenshtein
# alignment of 'hypothesis' and 'reference'
# (i.e. the output of 'align-text'.

# The information added to each token in the CTM is the reference word and one
# of the following edit-types:
#  'cor' = correct  [note: as a special case we count as correct cases where
#                    the hypothesis word is the OOV symbol and the reference
#                    word is OOV w.r.t. the supplied vocabulary.]
#  'sub' = substitution
#  'del' = deletion
#  'ins' = insertion
#  'sil' = (silence in ctm; does not consume a reference word)
# note: the script modify_ctm_edits.py will add the new
# note: the following extra edit-type may be added by modify_ctm_edits.py:
#  'fix'  ... this is like 'cor', but it means the reference has been modified
#             to fix non-scoreable errors [typically errors that don't change the
#             meaning], so we don't trust the word or value it as much as a 'cor'.
#

# Note: Additional lines are added to the CTM to account for deletions.

# Input CTM:
# (note: the <eps> is for silence in the input CTM that comes from
# optional-silence in the graph.  However, the input edits don't have anything
# for these silences.
# We assume (and check) that the channel will always be '1', because the
# input CTMs are expected to be 'per utterance', not including real
# recording-ids.

# Input ctm format:
# <file-id> <channel> <start-time> <duration> <hyp-word> [<confidence>]
# note, the confidence defaults to 1 if not provided (these
# scripts don't actually use the confidence field).

## TimBrown_2008P-0007226-0007620 1 0.000 0.100 when
## TimBrown_2008P-0007226-0007620 1 0.100 0.090 i
## TimBrown_2008P-0007226-0007620 1 0.190 0.300 some
## TimBrown_2008P-0007226-0007620 1 0.490 0.110 when
## TimBrown_2008P-0007226-0007620 1 0.600 0.060 i
## TimBrown_2008P-0007226-0007620 1 0.660 0.190 say
## TimBrown_2008P-0007226-0007620 1 0.850 0.450 go
## TimBrown_2008P-0007226-0007620 1 1.300 0.310 [COUGH]
## TimBrown_2008P-0007226-0007620 1 1.610 0.130 you
## TimBrown_2008P-0007226-0007620 1 1.740 0.180 got
## TimBrown_2008P-0007226-0007620 1 1.920 0.370 thirty
## TimBrown_2008P-0007226-0007620 1 2.290 0.830 seconds
## TimBrown_2008P-0007226-0007620 1 3.120 0.330 <eps>
## TimBrown_2008P-0007226-0007620 1 3.450 0.040 [BREATH]
## TimBrown_2008P-0007226-0007620 1 3.490 0.110 to
## TimBrown_2008P-0007226-0007620 1 3.600 0.320 [NOISE]

# Input Levenshtein edits : (the output of 'align-text' post-processed by 'wer_per_utt_details.pl')

# AJJacobs_2007P-0001605-0003029 i i ; thought thought ; i'd i'd ; tell tell ; you you ; a a ; little little ; about about ; [UH] [UH] ; what what ; i i ; like like ; to to ; write write ; and and ; [UH] [UH] ; i i ; like like ; to to ; [UH] [UH] ; immerse immerse ; myself myself ; [SMACK] [SMACK] ; in in ; my my ; topics topics ; [UM] [UM] ; i i ; just just ; like like ; to to ; [UH] [UH] ; dive dive ; [SMACK] [SMACK] ; right right ; in in ; and and ; become become ; [UH] [UH] ; sort sort ; of of ; a a ; human human ; guinea guinea ; pig pig ; [BREATH] [BREATH] ; and and ; [UH] [UH]
# AJJacobs_2007P-0003133-0004110 i i ; see see ; my my ; life life ; as as ; a a ; series series ; of of ; experiments experiments ; [BREATH] [BREATH] ; so so ; [UH] [UH] ; i i ; [NOISE] [NOISE] ; work work ; for for ; esquire esquire ; magazine magazine ; <eps> and ; a a ; couple couple ; of of ; years years ; ago ago ; [BREATH] [BREATH] ; i i ; wrote wrote ; an an ; article article ; called called ; [NOISE] [NOISE] ; my my ; outsourced outsourced ; life life


# Output format:
# <file-id> <channel> <start-time> <duration> <hyp-word> <confidence> <ref-word> <edit-type>

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
# AJJacobs_2007P-0001605-0003029 1 2.23 0.34 <eps> 1.0 <eps> sil
# AJJacobs_2007P-0001605-0003029 1 2.57 0.21 what 1.0 what cor
# AJJacobs_2007P-0001605-0003029 1 2.78 0.1 i 1.0 i cor
# AJJacobs_2007P-0001605-0003029 1 2.88 0.22 like 1.0 like cor
# AJJacobs_2007P-0001605-0003029 1 3.1 0.13 to 1.0 to cor
# AJJacobs_2007P-0001605-0003029 1 3.23 0.37 write 1.0 write cor
# AJJacobs_2007P-0001605-0003029 1 3.6 0.03 <eps> 1.0 <eps> sil
# AJJacobs_2007P-0001605-0003029 1 3.63 0.36 and 1.0 and cor



parser = argparse.ArgumentParser(
    description = "Append to the CTM the Levenshtein alignment of 'hypothesis' and 'reference'; "
    "creates augmented CTM with extra fields (see script for details)")

parser.add_argument("--oov", type = int, default = -1,
                    help = "The integer representation of the OOV symbol; substitutions "
                    "by the OOV symbol for out-of-vocabulary reference words are treated "
                    "as correct, if you also supply the --symbol-table option.")
parser.add_argument("--symbol-table", type = str,
                    help = "The words.txt your system used; if supplied, it is used to "
                    "determine OOV words (and such words will count as correct if "
                    "substituted by the OOV symbol).  See also the --oov option")
# Required arguments
parser.add_argument("edits_in", metavar = "<edits-in>",
                    help = "Filename of output of 'align-text', which this program reads. "
                    "Use /dev/stdin for standard input.")
parser.add_argument("ctm_in", metavar = "<ctm-in>",
                    help = "Filename of input hypothesis in ctm format")
parser.add_argument("ctm_edits_out", metavar = "<ctm-edits-out>",
                    help = "Filename of output (CTM appended with word-edit information)")
args = parser.parse_args()



def OpenFiles():
    global ctm_edits_out, edits_in, ctm_in, symbol_table, oov_word
    try:
        ctm_edits_out = open(args.ctm_edits_out, 'w')
    except:
        sys.exit("get_ctm_edits.py: error opening ctm-edits file {0} for output".format(
                args.ctm_edits_out))
    try:
        edits_in = open(args.edits_in)
    except:
        sys.exit("get_ctm_edits.py: error opening edits file {0} for input".format(
                args.edits_in))
    try:
        ctm_in = open(args.ctm_in)
    except:
        sys.exit("get_ctm_edits.py: error opening ctm file {0} for input".format(
                args.ctm_in))

    symbol_table = set()
    oov_word = None
    if args.symbol_table != None:
        if args.oov == -1:
            print("get_ctm_edits.py: error: if you set the the --symbol-table option "
                  "you must also set the --oov option", file = sys.stderr)
        try:
            f = open(args.symbol_table, 'r')
            for line in f.readlines():
                [ word, integer ] = line.split()
                if int(integer) == args.oov:
                    oov_word = word
                symbol_table.add(word)
        except:
            sys.exit("get_ctm_edits.py: error opening symbol-table file {0} for "
                     "input (or bad file), exception is: {1}".format(args.symbol_table))
        f.close()
        if oov_word == None:
            sys.exit("get_ctm_edits.py: OOV word not found: check the values of "
                     "--symbol-table={0} and --oov={1}".format(args.symbol_table,
                                                               args.oov))

# This function takes two lists
# edits_array = [ [ hyp_word1, ref_word1], [ hyp_word2, ref_word2 ], ... ]
# ctm_array = [ [ start1, duration1, hyp_word1, confidence1 ], ... ]
#
# and pads them with new list elements so that the entries 'match up'.  What we
# are aiming for is that for each i, ctm_array[i][2] == edits_array[i][0].  The
# reasons why this is not automatically true are:
#
#  (1) There may be deletions in the hypothesis sequence, which would lead to
#      pairs like [ '<eps>', ref_word ].
#  (2) The ctm may have been written 'with silence', which will lead to
#      ctm entries like [ 1, 7.8, 0.9, '<eps>' ] where the '<eps>' refers
#      to the optional-silence from the lexicon.
#
# We introduce suitable entries in to edits_array and ctm_array as necessary
# to make them 'match up'.  This function returns the pair (new_edits_array,
# new_ctm_array).
def PadArrays(edits_array, ctm_array):
    new_edits_array = []
    new_ctm_array = []
    edits_len = len(edits_array)
    ctm_len = len(ctm_array)
    edits_pos = 0
    ctm_pos = 0
    # current_time is the end of the last ctm segment we processesed.
    current_time = ctm_array[0][0] if ctm_len > 0 else 0.0
    while edits_pos < edits_len or ctm_pos < ctm_len:
        if edits_pos < edits_len and ctm_pos < ctm_len and \
                edits_array[edits_pos][0] == ctm_array[ctm_pos][2] and \
                edits_array[edits_pos][0] != '<eps>':
            # This is the normal case, where there are 2 entries where
            # they hyp-words match up
            new_edits_array.append(edits_array[edits_pos])
            edits_pos += 1
            new_ctm_array.append(ctm_array[ctm_pos])
            current_time = ctm_array[ctm_pos][0] + ctm_array[ctm_pos][1]
            ctm_pos += 1
        elif edits_pos < edits_len and edits_array[edits_pos][0] == '<eps>':
            # There was a deletion.  Pad with an empty ctm segment with '<eps>' as
            # the word.
            new_edits_array.append(edits_array[edits_pos])
            edits_pos += 1
            duration = 0.0
            confidence = 1.0
            new_ctm_array.append([ current_time, duration, '<eps>', confidence])
        elif ctm_pos < ctm_len and ctm_array[ctm_pos][2] == '<eps>':
            # There was silence in the ctm, and either we're reached the end of the
            # edits sequence, or the hyp word was not '<eps>':

            new_edits_array.append(['<eps>', '<eps>'])
            new_ctm_array.append(ctm_array[ctm_pos])
            current_time = ctm_array[ctm_pos][0] + ctm_array[ctm_pos][1]
            ctm_pos += 1
        else:
            raise Exception("Could not align edits_array = {0} and ctm_array = {1}; "
                            "edits-position = {2}, ctm-position = {3}, "
                            "pending-edit={4}, pending-ctm-entry={5}".format(
                    edits_array, ctm_array, edits_pos, ctm_pos,
                    edits_array[edits_pos] if edits_pos < edits_len else None,
                    ctm_array[ctm_pos] if ctm_pos < ctm_len else None))
    assert len(new_edits_array) == len(new_ctm_array)
    return (new_edits_array, new_ctm_array)


# This function returns the appropriate edit-type to output in the ctm-edits
# file.  The ref_word and hyp_word and duration are the values we'll print in
# the ctm-edits file.
def GetEditType(hyp_word, ref_word, duration):
    global oov_word
    if hyp_word == ref_word and hyp_word !='<eps>':
        return 'cor'
    elif hyp_word != '<eps>' and ref_word == '<eps>':
        return 'ins'
    elif hyp_word == '<eps>' and ref_word != '<eps>' and duration == 0.0:
        return 'del'
    elif hyp_word == oov_word and \
         len(symbol_table) != 0 and not ref_word in symbol_table:
        return 'cor'   # this special case is treated as correct.
    elif hyp_word == '<eps>' == ref_word and duration > 0.0:
        # silence in hypothesis; we don't match this up with any reference word.
        return 'sil'
    else:
        # The following assertion is because, based on how PadArrays
        # works, we shouldn't hit this case.
        assert hyp_word != '<eps>' and ref_word != '<eps>'
        return 'sub'

# this prints a number with a certain number of digits after
# the point, while removing trailing zeros.
def FloatToString(f):
    num_digits = 6 # we want to print 6 digits after the zero
    g = f
    while abs(g) > 1.0:
        g *= 0.1
        num_digits += 1
    format_str = '%.{0}g'.format(num_digits)
    return format_str % f


def OutputCtm(utterance_id, edits_array, ctm_array):
    global ctm_edits_out
    # note: this function expects the padded entries created by PadARrays.
    assert len(edits_array) == len(ctm_array)
    channel = '1'  # this is hardcoded at both input and output, since this CTM
                   # doesn't really represent recordings, only utterances.
    for i in range(len(edits_array)):
        ( hyp_word, ref_word ) = edits_array[i]
        ( start_time, duration, hyp_word2, confidence ) = ctm_array[i]
        if not hyp_word == hyp_word2:
            print("Error producing output CTM for edit = {0} and ctm = {1}".format(
                    edits_array[i], ctm_array[i]), file = sys.stderr)
            sys.exit(1)
        assert hyp_word == hyp_word2
        edit_type = GetEditType(hyp_word, ref_word, duration)
        print(utterance_id, channel, FloatToString(start_time),
              FloatToString(duration), hyp_word, confidence, ref_word,
              edit_type, file = ctm_edits_out)


def ProcessOneUtterance(utterance_id, edits_line, ctm_lines):
    try:
        # Remove the utterance-id from the beginning of the edits line
        edits_fields = edits_line[len(utterance_id) + 1:]

        # e.g. if edits_fields is now 'i i ; see be ; my my ', edits_array will become
        #  [ ['i', 'i'], ['see', 'be'], ['my', 'my'] ]
        fields_split = edits_fields.split()
        first_fields, second_fields = fields_split[0::3], fields_split[1::3]
        if (
            len(first_fields) != len(second_fields) or
            (len(fields_split) >= 3 and set(fields_split[2::3]) != {';'})
        ):
            sys.exit("get_ctm_edits.py: could not make sense of edits line: " + edits_line)

        edits_array = list(zip(first_fields, second_fields))

        # ctm_array will now become something like [ ['1', '1.010', '0.240', 'little ' ], ... ]
        ctm_array = [ x.split() for x in ctm_lines ]
        ctm_array = []
        for line in ctm_lines:
            try:
                # Strip off the utterance-id and split the remaining fields
                # which should be: channel==1, start, dur, word, [confidence]
                a = line[len(utterance_id) + 1:].split()
                if len(a) == 4:
                    a.append(1.0)  # confidence defaults to 1.0.
                [ channel, start, dur, word, confidence ] = a
                if channel != '1':
                    raise Exception("Channel should be 1, got: " + channel)
                ctm_array.append([ float(start), float(dur), word, float(confidence) ])
            except Exception as e:
                sys.exit("get_ctm_edits.py: error procesing ctm line {0} "
                         "... exception is: {1} {2}".format(line, type(e), str(e)))
        # ctm_array will now be something like [ [ 1.010, 0.240, 'little ', 1.0 ], ... ]

        # The following call pads the edits and ctm arrays with appropriate
        # entries so that they have the same length and the elements 'match up'.
        (edits_array, ctm_array) = PadArrays(edits_array, ctm_array)
    except Exception as e:
        sys.exit("get_ctm_edits.py: error processing utterance {0}, error was: {1}".format(
                utterance_id, str(e)))
    OutputCtm(utterance_id, edits_array, ctm_array)

def ProcessData():
    num_utterances_processed = 0

    pending_ctm_line = ctm_in.readline()

    while True:
        this_edits_line = edits_in.readline()
        if this_edits_line == '':
            if pending_ctm_line != '':
                sys.exit("get_ctm_edits.py: edits_in input {0} ended before "
                         "ctm input was ended.  We processed {1} "
                         "utterances.".format(args.edits_in, num_utterances_processed))
            break
        a = this_edits_line.split()
        if len(a) == 0:
            sys.exit("get_ctm_edits.py: edits_input {0} had an empty line".format(
                    args.edits_in))
        utterance_id = a[0]
        utterance_id_len = len(utterance_id)
        this_utterance_ctm_lines = []
        while len(pending_ctm_line.strip()) > 0 and pending_ctm_line.split()[0] == utterance_id:
            this_utterance_ctm_lines.append(pending_ctm_line)
            pending_ctm_line = ctm_in.readline()
        ProcessOneUtterance(utterance_id, this_edits_line,
                            this_utterance_ctm_lines)
        num_utterances_processed += 1
    print("get_ctm_edits.py: processed {0} utterances".format(
            num_utterances_processed), file=sys.stderr)


OpenFiles()
ProcessData()

