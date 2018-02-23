#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This module aligns a hypothesis (CTM or text) with a reference to
find the best matching sub-sequence in the reference for the hypothesis
using Smith-Waterman like alignment.

e.g.: align_ctm_ref.py --hyp-format=CTM --ref=data/train/text --hyp=foo/ctm
        --output=foo/ctm_edits
"""

from __future__ import print_function
import argparse
import logging
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)

verbose_level = 0


def get_args():
    parser = argparse.ArgumentParser(description="""
    This module aligns a hypothesis (CTM or text) with a reference to find the
    best matching sub-sequence in the reference for the hypothesis using
    Smith-Waterman like alignment.

    e.g.: align_ctm_ref.py --align-full-hyp=false --hyp-format=CTM
    --reco2file-and-channel=data/foo/reco2file_and_channel --ref=data/train/text
    --hyp=foo/ctm --output=foo/ctm_edits
    """)

    parser.add_argument("--hyp-format", type=str, choices=["Text", "CTM"],
                        default="CTM",
                        help="Format used for the hypothesis")
    parser.add_argument("--reco2file-and-channel", type=argparse.FileType('r'),
                        help="""reco2file_and_channel file.
                        This will be used to match references that are usually
                        indexed by the recording-id with the CTM lines that have
                        file and channel. This option is typically not
                        required.""")
    parser.add_argument("--eps-symbol", type=str, default="-",
                        help="Symbol used to contain alignment "
                        "to empty symbol")
    parser.add_argument("--oov-word", type=str, default=None,
                        action=common_lib.NullstrToNoneAction,
                        help="Symbol of OOV word in hypothesis")
    parser.add_argument("--symbol-table", type=argparse.FileType('r'),
                        help="""Symbol table for words in vocabulary. Used
                        to determine if a word is a OOV or not""")

    parser.add_argument("--correct-score", type=int, default=1,
                        help="Score for correct matches")
    parser.add_argument("--substitution-penalty", type=int, default=1,
                        help="Penalty for substitution errors")
    parser.add_argument("--deletion-penalty", type=int, default=1,
                        help="Penalty for deletion errors")
    parser.add_argument("--insertion-penalty", type=int, default=1,
                        help="Penalty for insertion errors")

    parser.add_argument("--align-full-hyp", type=str,
                        action=common_lib.StrToBoolAction,
                        choices=["true", "false"], default=True,
                        help="""Align full hypothesis i.e. trackback from
                        the end to get the alignment. This is different
                        from the normal Smith-Waterman alignment, where the
                        traceback will be from the maximum score.""")

    parser.add_argument("--debug-only", type=str, default="false",
                        choices=["true", "false"],
                        help="Run test functions only")
    parser.add_argument("--verbose", type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help="Use larger value for more verbose logging.")

    parser.add_argument("--ref", dest='ref_in_file',
                        type=argparse.FileType('r'), required=True,
                        help="Reference text file")
    parser.add_argument("--hyp", dest='hyp_in_file', required=True,
                        type=argparse.FileType('r'),
                        help="Hypothesis text or CTM file")
    parser.add_argument("--output", dest='alignment_out_file', required=True,
                        type=argparse.FileType('w'),
                        help="""File to write output alignment.
                        If hyp-format=CTM, then the output is in the form of
                        CTM, but with two additional columns of Edit-type and
                        Reference-word matched to the hypothesis.""")

    args = parser.parse_args()

    args.debug_only = bool(args.debug_only == "true")

    global verbose_level
    verbose_level = args.verbose
    if args.verbose > 2:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return args


def read_text(text_file):
    """Reads a kaldi-format text file and yield elements of a dictionary
        { utterane_id : transcript (as a list of words) }

    The first-column of the text file is the utterance-id, which will be
    used as the key to index the dictionary elements.
    The remaining columns of the file are text of the transcript and they are
    returned as a list of words.
    """
    for line in text_file:
        parts = line.strip().split()
        if len(parts) < 1:
            raise RuntimeError(
                "Did not get enough columns; line {0} in {1}"
                "".format(line, text_file.name))
        elif len(parts) == 1:
            logger.warn("Empty transcript for utterance %s in %s", 
                        parts[0], text_file.name)
            yield parts[0], []
        else:
            yield parts[0], parts[1:]
    text_file.close()


def read_ctm(ctm_file, file_and_channel2reco=None):
    """Reads a CTM file and yields elements of a dictionary
        { utterance-id : CTM for the utterance },
    where CTM for the utterance is stored as a list of lines
    from a CTM correponding to the utterance.

    Note: *_reco in the variables usually correspond to utterances rather
    than recordings.
    """
    prev_reco = ""
    ctm_lines = []
    for line in ctm_file:
        try:
            parts = line.strip().split()
            parts[2] = float(parts[2])
            parts[3] = float(parts[3])

            if len(parts) == 5:
                parts.append(1.0)   # confidence defaults to 1.0.

            if len(parts) != 6:
                raise ValueError("CTM must have 6 fields.")

            if file_and_channel2reco is None:
                reco = parts[0]
                if parts[1] != '1':
                    raise ValueError("Channel should be 1, "
                                     "got {0}".format(parts[1]))
            else:
                reco = file_and_channel2reco[(parts[0], parts[1])]
            if prev_reco != "" and reco != prev_reco:
                # New recording
                yield prev_reco, ctm_lines
                ctm_lines = []
            ctm_lines.append(parts[2:])
            prev_reco = reco
        except Exception:
            logger.error("Error in processing CTM line {0}".format(line))
            raise
    if prev_reco != "" and len(ctm_lines) > 0:
        yield prev_reco, ctm_lines
    ctm_file.close()


def smith_waterman_alignment(ref, hyp, similarity_score_function,
                             del_score, ins_score,
                             eps_symbol="<eps>", align_full_hyp=True):
    """Does Smith-Waterman alignment of reference sequence and hypothesis
    sequence.
    This is a special case of the Smith-Waterman alignment that assumes that
    the deletion and insertion costs are linear with number of incorrect words.

    If align_full_hyp is True, then the traceback of the alignment
    is started at the end of the hypothesis. This is when we want the
    reference that aligns with the full hypothesis.
    This differs from the normal Smith-Waterman alignment, where the traceback
    is from the highest score in the alignment score matrix. This
    can be obtained by setting align_full_hyp as False. This gets only the
    sub-sequence of the hypothesis that best matches with a
    sub-sequence of the reference.

    Returns a list of tuples where each tuple has the format:
        (ref_word, hyp_word, ref_word_from_index, hyp_word_from_index,
         ref_word_to_index, hyp_word_to_index)
    """
    output = []

    ref_len = len(ref)
    hyp_len = len(hyp)

    bp = [[] for x in range(ref_len+1)]

    # Score matrix of size (ref_len + 1) x (hyp_len + 1)
    # The index m, n in this matrix corresponds to the score
    # of the best matching sub-sequence pair between reference and hypothesis
    # ending with the reference word ref[m-1] and hypothesis word hyp[n-1].
    # If align_full_hyp is True, then the hypothesis sub-sequence is from
    # the 0th word i.e. hyp[0].
    H = [[] for x in range(ref_len+1)]

    for ref_index in range(ref_len+1):
        if align_full_hyp:
            H[ref_index] = [-(hyp_len+2) for x in range(hyp_len+1)]
            H[ref_index][0] = 0
        else:
            H[ref_index] = [0 for x in range(hyp_len+1)]
        bp[ref_index] = [(0, 0) for x in range(hyp_len+1)]

        if align_full_hyp and ref_index == 0:
            for hyp_index in range(1, hyp_len+1):
                H[0][hyp_index] = H[0][hyp_index-1] + ins_score
                bp[ref_index][hyp_index] = (ref_index, hyp_index-1)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4}"
                    "".format(ref_index, hyp_index-1, ref_index, hyp_index,
                              H[ref_index][hyp_index]))

    max_score = -float("inf")
    max_score_element = (0, 0)

    for ref_index in range(1, ref_len+1):     # Reference
        for hyp_index in range(1, hyp_len+1):     # Hypothesis
            sub_or_ok = (H[ref_index-1][hyp_index-1]
                         + similarity_score_function(ref[ref_index-1],
                                                     hyp[hyp_index-1]))

            if ((not align_full_hyp and sub_or_ok > 0)
                    or (align_full_hyp
                        and sub_or_ok >= H[ref_index][hyp_index])):
                H[ref_index][hyp_index] = sub_or_ok
                bp[ref_index][hyp_index] = (ref_index-1, hyp_index-1)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4} ({5},{6})"
                    "".format(ref_index-1, hyp_index-1, ref_index, hyp_index,
                              H[ref_index][hyp_index],
                              ref[ref_index-1], hyp[hyp_index-1]))

            if H[ref_index-1][hyp_index] + del_score > H[ref_index][hyp_index]:
                H[ref_index][hyp_index] = H[ref_index-1][hyp_index] + del_score
                bp[ref_index][hyp_index] = (ref_index-1, hyp_index)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4}"
                    "".format(ref_index-1, hyp_index, ref_index, hyp_index,
                              H[ref_index][hyp_index]))

            if H[ref_index][hyp_index-1] + ins_score > H[ref_index][hyp_index]:
                H[ref_index][hyp_index] = H[ref_index][hyp_index-1] + ins_score
                bp[ref_index][hyp_index] = (ref_index, hyp_index-1)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4}"
                    "".format(ref_index, hyp_index-1, ref_index, hyp_index,
                              H[ref_index][hyp_index]))

            #if hyp_index == hyp_len and H[ref_index][hyp_index] >= max_score:
            if ((not align_full_hyp or hyp_index == hyp_len)
                    and H[ref_index][hyp_index] >= max_score):
                max_score = H[ref_index][hyp_index]
                max_score_element = (ref_index, hyp_index)

    ref_index, hyp_index = max_score_element
    score = max_score
    logger.debug("Alignment score: %s for (%d, %d)",
                 score, ref_index, hyp_index)

    while ((not align_full_hyp and score >= 0)
           or (align_full_hyp and hyp_index > 0)):
        try:
            prev_ref_index, prev_hyp_index = bp[ref_index][hyp_index]

            if ((prev_ref_index, prev_hyp_index) == (ref_index, hyp_index)
                    or (prev_ref_index, prev_hyp_index) == (0, 0)):
                ref_index, hyp_index = (prev_ref_index, prev_hyp_index)
                score = H[ref_index][hyp_index]
                break

            if (ref_index == prev_ref_index + 1
                    and hyp_index == prev_hyp_index + 1):
                # Substitution or correct
                output.append(
                    (ref[ref_index-1] if ref_index > 0 else eps_symbol,
                     hyp[hyp_index-1] if hyp_index > 0 else eps_symbol,
                     prev_ref_index, prev_hyp_index, ref_index, hyp_index))
            elif (prev_hyp_index == hyp_index):
                # Deletion
                assert prev_ref_index == ref_index - 1
                output.append(
                    (ref[ref_index-1] if ref_index > 0 else eps_symbol,
                     eps_symbol,
                     prev_ref_index, prev_hyp_index, ref_index, hyp_index))
            elif (prev_ref_index == ref_index):
                # Insertion
                assert prev_hyp_index == hyp_index - 1
                output.append(
                    (eps_symbol,
                     hyp[hyp_index-1] if hyp_index > 0 else eps_symbol,
                     prev_ref_index, prev_hyp_index, ref_index, hyp_index))
            else:
                raise RuntimeError

            ref_index, hyp_index = (prev_ref_index, prev_hyp_index)
            score = H[ref_index][hyp_index]
        except Exception:
            logger.error("Unexpected entry (%d,%d) -> (%d,%d), %s, %s",
                         prev_ref_index, prev_hyp_index, ref_index, hyp_index,
                         ref[prev_ref_index], hyp[prev_hyp_index])
            raise RuntimeError("Unexpected result: Bug in code!!")

    assert (align_full_hyp or score == 0)

    output.reverse()

    if verbose_level > 2:
        for ref_index in range(ref_len+1):
            for hyp_index in range(hyp_len+1):
                print ("{0} ".format(H[ref_index][hyp_index]), end='',
                       file=sys.stderr)
            print ("", file=sys.stderr)

    logger.debug("Aligned output:")
    logger.debug("  -  ".join(["({0},{1})".format(x[4], x[5])
                               for x in output]))
    logger.debug("REF: ")
    logger.debug("    ".join(str(x[0]) for x in output))
    logger.debug("HYP:")
    logger.debug("    ".join(str(x[1]) for x in output))

    return (output, max_score)


def print_alignment(recording, alignment, out_file_handle):
    out_text = [recording]
    for line in alignment:
        try:
            out_text.append(line[1])
        except Exception:
            logger.error("Something wrong with alignment. "
                         "Invalid line {0}".format(line))
            raise
    print (" ".join(out_text), file=out_file_handle)


def get_edit_type(hyp_word, ref_word, duration=-1, eps_symbol='<eps>',
                  oov_word=None, symbol_table=None):
    if hyp_word == ref_word and hyp_word != eps_symbol:
        return 'cor'
    if hyp_word != eps_symbol and ref_word == eps_symbol:
        return 'ins'
    if hyp_word == eps_symbol and ref_word != eps_symbol and duration == 0.0:
        return 'del'
    if (hyp_word == oov_word and symbol_table is not None
            and len(symbol_table) > 0 and ref_word not in symbol_table):
        return 'cor'    # this special case is treated as correct
    if hyp_word == eps_symbol and ref_word == eps_symbol and duration > 0.0:
        # silence in hypothesis; we don't match this up with any reference
        # word.
        return 'sil'
    # The following assertion is because, based on how get_ctm_edits()
    # works, we shouldn't hit this case.
    assert hyp_word != eps_symbol and ref_word != eps_symbol
    return 'sub'


def get_ctm_edits(alignment_output, ctm_array, eps_symbol="<eps>",
                  oov_word=None, symbol_table=None):
    """
    This function takes two lists
        alignment_output = The output of smith_waterman_alignment() which is a
            list of tuples (ref_word, hyp_word, ref_word_from_index,
            hyp_word_from_index, ref_word_to_index, hyp_word_to_index)
        ctm_array = [ [ start1, duration1, hyp_word1, confidence1 ], ... ]
    and pads them with new list elements so that the entries 'match up'.

    Returns CTM edits lines, which are CTM lines appended with reference word
    and edit type.

    What we are aiming for is that for each i, ctm_array[i][2] ==
    alignment_output[i][1].  The reasons why this is not automatically true
    are:

     (1) There may be insertions in the hypothesis sequence that are not
         aligned with any reference words in the beginning of the
         alignment_output.
     (2) There may be deletions in the end of the alignment_output that
         do not correspond to any additional hypothesis CTM lines.

    We introduce suitable entries in to alignment_output and ctm_array as
    necessary to make them 'match up'.
    """
    ctm_edits = []
    ali_len = len(alignment_output)
    ctm_len = len(ctm_array)
    ali_pos = 0
    ctm_pos = 0

    # current_time is the end of the last ctm segment we processesed.
    current_time = ctm_array[0][0] if ctm_len > 0 else 0.0

    for (ref_word, hyp_word, ref_prev_i, hyp_prev_i,
         ref_i, hyp_i) in alignment_output:
        try:
            ctm_pos = hyp_prev_i
            # This is true because we cannot have errors at the end because
            # that will decrease the smith-waterman alignment score.
            assert ctm_pos < ctm_len
            assert len(ctm_array[ctm_pos]) == 4

            if hyp_prev_i == hyp_i:
                assert hyp_word == eps_symbol
                # These are deletions as there are no CTM entries
                # corresponding to these alignments.
                edit_type = get_edit_type(
                    hyp_word=eps_symbol, ref_word=ref_word,
                    duration=0.0, eps_symbol=eps_symbol,
                    oov_word=oov_word, symbol_table=symbol_table)
                ctm_line = [current_time, 0.0, eps_symbol, 1.0,
                            ref_word, edit_type]
                ctm_edits.append(ctm_line)
            else:
                assert hyp_i == hyp_prev_i + 1
                assert hyp_word == ctm_array[ctm_pos][2]
                # This is the normal case, where there are 2 entries where
                # they hyp-words match up.
                ctm_line = list(ctm_array[ctm_pos])
                if hyp_word == eps_symbol and ref_word != eps_symbol:
                    # This is a silence in hypothesis aligned with a reference
                    # word. We split this into two ctm edit lines where the
                    # first one is a deletion of duration 0 and the second
                    # one is a silence of duration given by the ctm line.
                    edit_type = get_edit_type(
                        hyp_word=eps_symbol, ref_word=ref_word,
                        duration=0.0, eps_symbol=eps_symbol,
                        oov_word=oov_word, symbol_table=symbol_table)
                    assert edit_type == 'del'
                    ctm_edits.append([current_time, 0.0, eps_symbol, 1.0,
                                      ref_word, edit_type])

                    edit_type = get_edit_type(
                        hyp_word=eps_symbol, ref_word=eps_symbol,
                        duration=ctm_line[1], eps_symbol=eps_symbol,
                        oov_word=oov_word, symbol_table=symbol_table)
                    assert edit_type == 'sil'
                    ctm_line.extend([eps_symbol, edit_type])
                    ctm_edits.append(ctm_line)
                else:
                    edit_type = get_edit_type(
                        hyp_word=hyp_word, ref_word=ref_word,
                        duration=ctm_line[1], eps_symbol=eps_symbol,
                        oov_word=oov_word, symbol_table=symbol_table)
                    ctm_line.extend([ref_word, edit_type])
                    ctm_edits.append(ctm_line)
                current_time = (ctm_array[ctm_pos][0]
                                + ctm_array[ctm_pos][1])
        except Exception:
            logger.error("Could not get ctm edits for "
                         "edits@{edits_pos} = {0}, ctm@{ctm_pos} = {1}".format(
                            ("NONE" if ali_pos >= ali_len
                             else alignment_output[ali_pos]),
                            ("NONE" if ctm_pos >= ctm_len
                             else ctm_array[ctm_pos]),
                            edits_pos=ali_pos, ctm_pos=ctm_pos))
            logger.error("alignment = {0}".format(alignment_output))
            raise
    return ctm_edits


def ctm_line_to_string(ctm_line):
    if len(ctm_line) != 8:
        raise RuntimeError("len(ctm_line) expected to be {0}. "
                           "Invalid line {1}".format(8, ctm_line))

    return " ".join([str(x) for x in ctm_line])


def test_alignment(align_full_hyp):
    hyp = "GCCAT"
    ref = "AGCACACA"

    verbose = 3
    logger.info("REF: %s", ref)
    logger.info("HYP: %s", hyp)

    output, score = smith_waterman_alignment(
        ref, hyp, similarity_score_function=lambda x, y: 2 if (x == y) else -1,
        del_score=-1, ins_score=-1, eps_symbol="-", align_full_hyp=align_full_hyp)

    print_alignment("Alignment", output, out_file_handle=sys.stderr)


def run(args):
    if args.debug_only:
        test_alignment(args.align_full_hyp)
        raise SystemExit("Exiting since --debug-only was true")

    def similarity_score_function(x, y):
        if x == y:
            return args.correct_score
        return -args.substitution_penalty

    del_score = -args.deletion_penalty
    ins_score = -args.insertion_penalty

    reco2file_and_channel = {}
    file_and_channel2reco = {}

    if args.reco2file_and_channel is not None:
        for line in args.reco2file_and_channel:
            parts = line.strip().split()

            reco2file_and_channel[parts[0]] = (parts[1], parts[2])
            file_and_channel2reco[(parts[1], parts[2])] = parts[0]
        args.reco2file_and_channel.close()
    else:
        file_and_channel2reco = None

    symbol_table = {}
    if args.symbol_table is not None:
        for line in args.symbol_table:
            parts = line.strip().split()
            symbol_table[parts[0]] = int(parts[1])
        args.symbol_table.close()

    if args.hyp_format == "Text":
        hyp_lines = {key: value
                     for (key, value) in read_text(args.hyp_in_file)}
    else:
        hyp_lines = {key: value
                     for (key, value) in read_ctm(args.hyp_in_file,
                                                  file_and_channel2reco)}

    num_err = 0
    num_done = 0
    for reco, ref_text in read_text(args.ref_in_file):
        try:
            if reco not in hyp_lines:
                num_err += 1
                raise Warning("Could not find recording {0} "
                              "in hypothesis {1}".format(
                                  reco, args.hyp_in_file.name))
                continue

            if args.hyp_format == "CTM":
                hyp_array = [x[2] for x in hyp_lines[reco]]
            else:
                hyp_array = hyp_lines[reco]

            if args.reco2file_and_channel is None:
                reco2file_and_channel[reco] = (reco, "1")

            logger.debug("Running Smith-Waterman alignment for %s", reco)

            output, score = smith_waterman_alignment(
                ref_text, hyp_array, eps_symbol=args.eps_symbol,
                similarity_score_function=similarity_score_function,
                del_score=del_score, ins_score=ins_score,
                align_full_hyp=args.align_full_hyp)

            if args.hyp_format == "CTM":
                ctm_edits = get_ctm_edits(output, hyp_lines[reco],
                                          eps_symbol=args.eps_symbol,
                                          oov_word=args.oov_word,
                                          symbol_table=symbol_table)
                for line in ctm_edits:
                    ctm_line = list(reco2file_and_channel[reco])
                    ctm_line.extend(line)
                    print(ctm_line_to_string(ctm_line),
                          file=args.alignment_out_file)
            else:
                print_alignment(
                    reco, output, out_file_handle=args.alignment_out_file)
            num_done += 1
        except:
            logger.error("Alignment failed for recording {0} "
                         "with ref = {1} and hyp = {2}".format(
                             reco, " ".join(ref_text),
                             " ".join(hyp_array)))
            raise

    logger.info("Processed %d recordings; failed with %d", num_done, num_err)

    if num_done == 0:
        raise RuntimeError("Processed 0 recordings.")


def main():
    args = get_args()

    try:
        run(args)
    except Exception:
        logger.error("Failed to align ref and hypotheses; "
                     "got exception ", exc_info=True)
        raise SystemExit(1)
    finally:
        if args.reco2file_and_channel is not None:
            args.reco2file_and_channel.close()
        args.ref_in_file.close()
        args.hyp_in_file.close()
        args.alignment_out_file.close()


if __name__ == '__main__':
    main()
