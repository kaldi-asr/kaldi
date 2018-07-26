#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This module aligns text from multiple files and outputs confusion network
as an FST in text format."""


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
    This module aligns text from multiple files and outputs confusion network
    as an FST in text format.
    """)

    parser.add_argument("--eps-symbol", type=str, default="<eps>",
                        help="Symbol used to contain alignment "
                        "to empty symbol")

    parser.add_argument("--correct-score", type=int, default=1,
                        help="Score for correct matches")
    parser.add_argument("--substitution-penalty", type=int, default=1,
                        help="Penalty for substitution errors")
    parser.add_argument("--deletion-penalty", type=int, default=1,
                        help="Penalty for deletion errors")
    parser.add_argument("--insertion-penalty", type=int, default=1,
                        help="Penalty for insertion errors")

    parser.add_argument("--verbose", type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help="Use larger value for more verbose logging.")

    parser.add_argument("lang", type=str, help="Lang directory")
    parser.add_argument("text_files", nargs='+', type=str,
                        help="Text files to combine")
    parser.add_argument("graph_file", type=argparse.FileType('w'),
                        help="""File to write output graph.""")

    args = parser.parse_args()

    global verbose_level
    verbose_level = args.verbose
    if args.verbose > 2:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return args


def levenshtein_alignment(ref, hyp, similarity_score_function,
                          del_score, ins_score,
                          eps_symbol="<eps>"):
    output = []

    ref_len = len(ref)
    hyp_len = len(hyp)

    bp = [[] for x in range(ref_len+1)]

    # Score matrix of size (ref_len + 1) x (hyp_len + 1)
    # The index m, n in this matrix corresponds to the score
    # of the best matching sub-sequence pair between reference and hypothesis
    # ending with the reference word ref[m-1] and hypothesis word hyp[n-1].
    H = [[] for x in range(ref_len+1)]

    # Initialize scores.
    # H[0][0] is set to 0 with everything -"inf"
    for ref_index in range(ref_len+1):
        H[ref_index] = [-float("inf") for x in range(hyp_len+1)]
        bp[ref_index] = [(0, 0) for x in range(hyp_len+1)]

        H[0][0] = 0
        if ref_index == 0:
            for hyp_index in range(1, hyp_len+1):
                # Add insertion for considering each hyp word
                H[0][hyp_index] = H[0][hyp_index-1] + ins_score
                bp[0][hyp_index] = (ref_index, hyp_index-1)
        else:
            # Add deletion for considering each ref word
            H[ref_index][0] = H[ref_index-1][0] + del_score
            bp[ref_index][0] = (ref_index-1, 0)

    for ref_index in range(1, ref_len+1):   # Reference
        for hyp_index in range(1, hyp_len+1):   # Hypothesis
            # Consider a substitution or correct transition
            sub_or_ok = (H[ref_index-1][hyp_index-1]
                         + similarity_score_function(ref[ref_index-1],
                                                     hyp[hyp_index-1]))

            if sub_or_ok >= H[ref_index][hyp_index]:
                # Substitution
                H[ref_index][hyp_index] = sub_or_ok
                bp[ref_index][hyp_index] = (ref_index-1, hyp_index-1)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4} ({5},{6})"
                    "".format(ref_index-1, hyp_index-1, ref_index, hyp_index,
                              H[ref_index][hyp_index],
                              ref[ref_index-1], hyp[hyp_index-1]))

            if H[ref_index-1][hyp_index] + del_score > H[ref_index][hyp_index]:
                # Deletion
                H[ref_index][hyp_index] = H[ref_index-1][hyp_index] + del_score
                bp[ref_index][hyp_index] = (ref_index-1, hyp_index)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4}"
                    "".format(ref_index-1, hyp_index, ref_index, hyp_index,
                              H[ref_index][hyp_index]))

            if H[ref_index][hyp_index-1] + ins_score > H[ref_index][hyp_index]:
                # Insertion
                H[ref_index][hyp_index] = H[ref_index][hyp_index-1] + ins_score
                bp[ref_index][hyp_index] = (ref_index, hyp_index-1)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4}"
                    "".format(ref_index, hyp_index-1, ref_index, hyp_index,
                              H[ref_index][hyp_index]))

    # Traceback from the corner
    ref_index = ref_len
    hyp_index = hyp_len
    score = H[ref_index][hyp_index]

    while hyp_index > 0 or ref_index > 0:
        try:
            prev_ref_index, prev_hyp_index = bp[ref_index][hyp_index]

            if (ref_index == prev_ref_index + 1
                    and hyp_index == prev_hyp_index + 1):
                # Substitution or correct
                output.append(
                    (ref[ref_index-1] if ref_index > 0 else eps_symbol,
                     hyp[hyp_index-1] if hyp_index > 0 else eps_symbol))
            elif prev_hyp_index == hyp_index:
                # Deletion
                assert prev_ref_index == ref_index - 1
                output.append(
                    (ref[ref_index-1] if ref_index > 0 else eps_symbol,
                     eps_symbol))
            elif prev_ref_index == ref_index:
                # Insertion
                assert prev_hyp_index == hyp_index - 1
                output.append(
                    (eps_symbol,
                     hyp[hyp_index-1] if hyp_index > 0 else eps_symbol))
            else:
                raise RuntimeError

            ref_index, hyp_index = (prev_ref_index, prev_hyp_index)
            if (ref_index, hyp_index) == (0, 0):
                break
        except Exception:
            logger.error("Unexpected entry (%d,%d) -> (%d,%d), %s, %s",
                         prev_ref_index, prev_hyp_index, ref_index, hyp_index,
                         ref[prev_ref_index], hyp[prev_hyp_index])
            raise RuntimeError("Unexpected result: Bug in code!!")

    assert ref_index == 0 and hyp_index == 0

    output.reverse()
    if verbose_level > 2:
        for ref_index in range(ref_len+1):
            for hyp_index in range(hyp_len+1):
                print ("{0} ".format(H[ref_index][hyp_index]), end='',
                       file=sys.stderr)
            print ("", file=sys.stderr)

    logger.debug("REF: ")
    logger.debug("    ".join(str(x[0]) for x in output))
    logger.debug("HYP:")
    logger.debug("    ".join(str(x[1]) for x in output))

    return (output, score)


def linear_to_graph(ref, symbol_table, oov_word, eps_symbol="<eps>"):
    lines = []
    for i, word in enumerate(ref):
        if word == eps_symbol:
            label = 0
        elif word not in symbol_table:
            label = symbol_table[oov_word]
        else:
            label = symbol_table[word]
        lines.append("{state} {next_state} {label} {label}"
                     "".format(state=i, next_state=i + 1, label=label))
    lines.append("{0}".format(len(ref)))
    return "\n".join(lines)


def sausage_to_graph(sausage, symbol_table, oov_word, eps_symbol="<eps>"):
    lines = []
    for i, link in enumerate(sausage):
        for word in link:
            if word == eps_symbol:
                label = 0
            elif word not in symbol_table:
                label = symbol_table[oov_word]
            else:
                label = symbol_table[word]
            lines.append("{state} {next_state} {label} {label}"
                         "".format(state=i, next_state=i + 1, label=label))
    lines.append("{0}".format(len(sausage)))
    return "\n".join(lines)


def add_to_sausage(output, sausage, eps_symbol="<eps>"):
    if len(sausage) == 0:
        for pair in output:
            ref, hyp = pair
            sausage.append([ref, hyp])
        return sausage

    assert len(sausage) >= len(output)

    sausage_idx = 0
    output_idx = 0

    while output_idx < len(output):
        ref, hyp = output[output_idx]
        if sausage_idx == len(sausage):
            assert ref == eps_symbol
            sausage.append([hyp for i in range(len(sausage[-1]))])
            output_idx += 1
        else:
            assert len(sausage[sausage_idx]) >= 2, sausage
            if ref == sausage[sausage_idx][0]:
                sausage[sausage_idx].append(hyp)
                sausage_idx += 1
                output_idx += 1
            elif ref == eps_symbol:
                sausage[sausage_idx].append(hyp)
                output_idx += 1
            elif sausage[sausage_idx][0] == eps_symbol:
                sausage_idx += 1
            else:
                raise RuntimeError("Unexpected pair {0}; sausage = {1}, "
                                   "output = {2}".format(
                                       output[output_idx], sausage, output))

    while sausage_idx < len(sausage):
        sausage[sausage_idx].append(eps_symbol)
        sausage_idx += 1

    return sausage


def run(args):
    def similarity_score_function(x, y):
        if x == y:
            return args.correct_score
        return -args.substitution_penalty

    del_score = -args.deletion_penalty
    ins_score = -args.insertion_penalty


    with open("{0}/oov.txt".format(args.lang)) as f:
        oov_word = f.readline().strip()
        assert len(oov_word.split(" ")) == 1

    symbol_table = {}
    with open("{0}/words.txt".format(args.lang)) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                raise RuntimeError("Unable to parser line {0}".format(line))
            symbol_table[parts[0]] = int(parts[1])

    file_handles = []
    for f in args.text_files:
        file_handle = open(f, 'r')
        file_handles.append(file_handle)

    current_utts = ["" for i in range(len(file_handles))]
    num_done = 0
    for line in file_handles[0]:
        parts = line.strip().split()
        utt_id = parts[0]
        ref1 = parts[1:]

        current_utts[0] = utt_id

        current_alignments = []
        max_len = 0
        idx_max_len = -1

        for i, this_file in enumerate(file_handles[1:]):
            i += 1
            while current_utts[i] == "" or current_utts[i] < utt_id:
                this_line = this_file.readline()
                parts = this_line.strip().split()
                current_utts[i] = parts[0]

            if current_utts[i] == utt_id:
                this_ref = parts[1:]

                output, score = levenshtein_alignment(
                    ref1, this_ref, eps_symbol=args.eps_symbol,
                    similarity_score_function=similarity_score_function,
                    del_score=del_score, ins_score=ins_score)

                current_alignments.append(output)
                if len(output) > max_len:
                    max_len = len(output)
                    idx_max_len = len(current_alignments) - 1

        if max_len == 0:
            assert len(current_alignments) == 0
            print ("{0}\n{1}\n".format(utt_id, linear_to_graph(
                ref1, symbol_table, oov_word, args.eps_symbol)),
                   file=args.graph_file)
        else:
            sausage = []
            output = current_alignments[idx_max_len]

            add_to_sausage(output, sausage)

            for x, output in enumerate(current_alignments):
                if x == idx_max_len:
                    continue
                else:
                    add_to_sausage(output, sausage)
            print ("{0}\n{1}\n".format(utt_id, sausage_to_graph(
                        sausage, symbol_table, oov_word, args.eps_symbol)),
                   file=args.graph_file)
        num_done += 1

    for f in file_handles:
        f.close()
    args.graph_file.close()

    logger.info("Processed %d utterances", num_done)

    if num_done == 0:
        raise RuntimeError("Processed 0 recordings.")


def main():
    args = get_args()

    try:
        run(args)
    except Exception:
        logger.error("Encountered failure!")
        raise

if __name__ == '__main__':
    main()
