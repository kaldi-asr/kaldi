#! /usr/bin/env python

from __future__ import print_function
import argparse
import logging
import sys

sys.path.insert(0, 'steps')
import libs.exceptions as kaldi_exceptions

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)


def _get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hyp-format", type=str, choices=["Text", "CTM"],
                        default="CTM",
                        help="Format used for the hypothesis")
    parser.add_argument("--reco2file-and-channel", type=argparse.FileType('r'),
                        help="reco2file_and_channel file, "
                        "required for CTM format hypothesis")
    parser.add_argument("--eps-symbol", type=str, default="-",
                        help="Symbol used to contain alignment "
                        "to empty symbol")
    parser.add_argument("--oov-word", type=str, default=None,
                        help="Symbol of OOV word in hypothesis")
    parser.add_argument("--symbol-table", type=argparse.FileType('r'),
                        help="Symbol table for words in vocabulary.")

    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--correct-score", type=int, default=1,
                        help="Score for correct matches")
    parser.add_argument("--substitution-penalty", type=int, default=1,
                        help="Penalty for substitution errors")
    parser.add_argument("--deletion-penalty", type=int, default=1,
                        help="Penalty for deletion errors")
    parser.add_argument("--insertion-penalty", type=int, default=1,
                        help="Penalty for insertion errors")
    parser.add_argument("--debug-only", type=str, default="false",
                        choices=["true", "false"],
                        help="Run test functions only")
    parser.add_argument("--ref", dest='ref_in_file',
                        type=argparse.FileType('r'), required=True,
                        help="Reference text file")
    parser.add_argument("--hyp", dest='hyp_in_file', required=True,
                        type=argparse.FileType('r'),
                        help="Hypothesis text or CTM file")
    parser.add_argument("--output", dest='alignment_out_file', required=True,
                        type=argparse.FileType('w'),
                        help="File to write output alignment.")

    args = parser.parse_args()

    if args.hyp_format == "CTM" and args.reco2file_and_channel is None:
        raise kaldi_exceptions.ArgumentError(
            "--reco2file-and-channel must be provided for "
            "hyp-format=CTM")

    args.debug_only = bool(args.debug_only == "true")

    if args.verbose > 2:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return args


def read_text(text_file):
    for line in text_file:
        parts = line.strip().split()
        if len(parts) <= 2:
            raise kaldi_exceptions.InputError(
                "Did not get enough columns.",
                line=line, input_file=text_file.name)
        yield parts[0], parts[1:]
    text_file.close()


def read_ctm(ctm_file, file_and_channel2reco=None):
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
                             eps_symbol="<eps>"):
    """Does Smith-Waterman alignment of reference sequence and hypothesis
    sequence.
    This is a special case of the Smith-Waterman alignment that assumes that
    the deletion and insertion costs are linear with number of incorrect words.

    Returns a list of tuples where each tuple has the format:
        (ref_word, hyp_word, ref_word_from_index, hyp_word_from_index,
         ref_word_to_index, hyp_word_to_index)
    """

    output = []

    M = len(ref)
    N = len(hyp)

    bp = [[] for x in range(M+1)]
    H = [[] for x in range(M+1)]

    for m in range(M+1):
        H[m] = [0 for x in range(N+1)]
        bp[m] = [(0, 0) for x in range(N+1)]

    max_score = 0
    max_score_element = (0, 0)

    for m in range(1, M+1):
        for n in range(1, N+1):
            sub_or_ok = (H[m-1][n-1]
                         + similarity_score_function(ref[m-1], hyp[n-1]))

            if sub_or_ok > 0:
                H[m][n] = sub_or_ok
                bp[m][n] = (m-1, n-1)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4} ({5},{6})".format(
                        m-1, n-1, m, n, H[m][n], ref[m-1], hyp[n-1]))

            if H[m-1][n] + del_score > H[m][n]:
                H[m][n] = H[m-1][n] + del_score
                bp[m][n] = (m-1, n)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4}".format(m-1, n, m, n,
                                                         H[m][n]))

            if H[m][n-1] + ins_score > H[m][n]:
                H[m][n] = H[m][n-1] + ins_score
                bp[m][n] = (m, n-1)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4}".format(m, n-1, m, n,
                                                         H[m][n]))

            if n == N and H[m][n] > max_score:
                max_score = H[m][n]
                max_score_element = (m, n)

    m, n = max_score_element
    score = max_score

    while score > 0:
        prev_m, prev_n = bp[m][n]

        if (m == prev_m + 1 and n == prev_n + 1):
            # Substitution or correct
            output.append((ref[m-1] if m > 0 else eps_symbol,
                           hyp[n-1] if n > 0 else eps_symbol,
                           prev_m, prev_n, m, n))
        elif (prev_n == n):
            # Deletion
            assert prev_m == m - 1
            output.append((ref[m-1] if m > 0 else eps_symbol,
                           eps_symbol,
                           prev_m, prev_n, m, n))
        elif (prev_m == m):
            # Insertion
            assert prev_n == n - 1
            output.append((eps_symbol,
                           hyp[n-1] if n > 0 else eps_symbol,
                           prev_m, prev_n, m, n))
        else:
            if (prev_m, prev_n) != 0:
                raise Exception("Unexpected result: Bug in code!!")

        m, n = (prev_m, prev_n)
        score = H[m][n]

    assert(score == 0)

    output.reverse()

    for m in range(M+1):
        for n in range(N+1):
            logger.debug("{0} ".format(H[m][n]))
        logger.debug("")

    logger.debug("\t-\t".join(["({0},{1})".format(x[4], x[5])
                               for x in output]))
    logger.debug("\t\t".join(str(x[0]) for x in output))
    logger.debug("\t\t".join(str(x[1]) for x in output))
    logger.debug(str(score))

    return (output, max_score)


def print_alignment(recording, alignment, eps_symbol, out_file_handle):
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
    if (hyp_word == oov_word
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


def test_alignment():
    hyp = "ACACACTA"
    ref = "AGCACACA"

    output, score = smith_waterman_alignment(
        ref, hyp, similarity_score_function=lambda x, y: 2 if (x == y) else -1,
        del_score=-1, ins_score=-1, eps_symbol="-")

    print_alignment("Alignment", output, eps_symbol="-",
                    out_file_handle=sys.stderr)


def _run(args):
    if args.debug_only:
        test_alignment()
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
                reco2file_and_channel[reco] = "1"

            output, score = smith_waterman_alignment(
                ref_text, hyp_array, eps_symbol=args.eps_symbol,
                similarity_score_function=similarity_score_function,
                del_score=del_score, ins_score=ins_score)

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
                    reco, output, eps_symbol=args.eps_symbol,
                    out_file_handle=args.alignment_out_file)
        except:
            logger.error("Alignment failed for recording {0} "
                         "with ref = {1} and hyp = {2}".format(
                             reco, " ".join(ref_text),
                             " ".join(hyp_array)))
            raise


def main():
    args = _get_args()

    try:
        _run(args)
    finally:
        if args.reco2file_and_channel is not None:
            args.reco2file_and_channel.close()
        args.ref_in_file.close()
        args.hyp_in_file.close()
        args.alignment_out_file.close()


if __name__ == '__main__':
    main()
