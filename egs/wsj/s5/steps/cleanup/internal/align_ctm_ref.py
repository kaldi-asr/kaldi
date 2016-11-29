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
    parser.add_argument("ref_in_file", type=argparse.FileType('r'),
                        help="Reference text file")
    parser.add_argument("hyp_in_file", type=argparse.FileType('r'),
                        help="Hypothesis text or CTM file")
    parser.add_argument("alignment_out_file", type=argparse.FileType('w'),
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


def smith_waterman_alignment(ref, hyp, eps_symbol,
                             similarity_score, del_score, ins_score,
                             hyp_format="CTM", file_=None, channel="1"):
    output = []

    if hyp_format == "CTM" and file_ is None:
        raise kaldi_exceptions.ArgumentError(
            "file_ is required if hyp_format is CTM")

    M = len(ref)
    N = len(hyp)

    bp = [[] for x in range(M+1)]
    H = [[] for x in range(M+1)]

    for m in range(M+1):
        H[m] = [0 for x in range(N+1)]
        bp[m] = [(0, 0) for x in range(N+1)]

    global_max = 0
    global_max_element = (0, 0)

    for m in range(1, M+1):
        for n in range(1, N+1):
            sub_or_ok = (H[m-1][n-1] +
                         similarity_score(ref[m-1],
                                          hyp[n-1][4]
                                          if hyp_format == "CTM"
                                          else hyp[n-1]))
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

            if H[m][n] > global_max:
                global_max = H[m][n]
                global_max_element = (m, n)

    m, n = global_max_element
    score = global_max

    # output.append( (m, n,
    #                 ref[m-1] if m > 0 else eps_symbol,
    #                 hyp[n-1] if n > 0 else eps_symbol) )

    while score > 0:
        prev_m, prev_n = bp[m][n]

        if (m == prev_m + 1 and n == prev_n + 1):
            # Substitution or correct
            output.append((ref[m-1] if m > 0 else eps_symbol,
                           hyp[n-1] if n > 0 else eps_symbol,
                           prev_m, prev_n, m, n))
        elif (prev_n == n):
            # Deletion
            fake_hyp = eps_symbol
            if hyp_format == "CTM":
                if n > 0:
                    fake_hyp = []
                    fake_hyp.extend(hyp[n-1])
                    assert len(hyp[n-1]) in [5, 6]
                    assert fake_hyp[0] == file_ and fake_hyp[1] == channel
                    fake_hyp[2] += fake_hyp[3]
                    fake_hyp[3] = 0.0
                    fake_hyp[4] = eps_symbol
                else:
                    fake_hyp = (file_, channel, 0, 0, eps_symbol)
            output.append((ref[m-1] if m > 0 else eps_symbol,
                           fake_hyp,
                           prev_m, prev_n, m, n))
        elif (prev_m == m):
            # Insertion
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

    return (output, global_max)


def test_alignment():
    hyp = "ACACACTA"
    ref = "AGCACACA"

    output, score = smith_waterman_alignment(
        ref, hyp, "-",
        lambda x, y: 2 if (x == y) else -1,
        -1, -1, hyp_format="Text")

    print_alignment("Alignment", dict(), output, "-", "Text",
                    sys.stderr)


def read_text(text_file):
    for line in text_file:
        parts = line.strip().split()
        if len(parts) <= 2:
            raise kaldi_exceptions.InputError(
                "Did not get enough columns.",
                line=line, input_file=text_file.name)
        yield parts[0], parts[1:]
    text_file.close()


def read_ctm(ctm_file, file_and_channel2reco):
    all_ctm = {}
    prev_reco = ""
    ctm_lines = []
    for line in ctm_file:
        try:
            parts = line.strip().split()
            parts[2] = float(parts[2])
            parts[3] = float(parts[3])
            reco = file_and_channel2reco[(parts[0], parts[1])]
            if prev_reco != "" and reco != prev_reco:
                # New recording
                all_ctm[prev_reco] = ctm_lines
                ctm_lines = []
            ctm_lines.append(parts)
            prev_reco = reco
        except Exception as e:
            logger.error(e, exc_info=True)
            raise kaldi_exceptions.InputError(str(e), line=line)
    if prev_reco != "" and len(ctm_lines) > 0:
        all_ctm[prev_reco] = ctm_lines
    ctm_file.close()
    return all_ctm


def print_alignment(recording, reco2file_and_channel,
                    alignment, eps_symbol,
                    hyp_format, out_file_handle):

    out_text = [recording]
    for line in alignment:
        try:
            ref_word = line[0]
            if hyp_format == "CTM":
                hyp_entry = line[1]

                current_time = hyp_entry[2] + hyp_entry[3]

                if hyp_entry == eps_symbol:
                    file_, channel = reco2file_and_channel[recording]
                    hyp_entry = (file_, channel, current_time, 0, eps_symbol,
                                 1.0, ref_word, "del")
                else:
                    if len(hyp_entry) == 5:
                        hyp_entry.append(1.0)
                    if (ref_word == eps_symbol):
                        error_type = "ins"
                    elif (ref_word != hyp_entry[4]):
                        error_type = "sub"
                    else:
                        error_type = "cor"
                    hyp_entry.extend([ref_word, error_type])

                hyp_entry = ctm_line_to_string(hyp_entry)
                print (hyp_entry, file=out_file_handle)
            else:
                out_text.append(line[1])
        except Exception as e:
            logger.error(e, exc_info=True)
            raise RuntimeError("Something wrong with alignment. "
                               "Invalid line {0}".format(line))
    if hyp_format != "CTM":
        print (" ".join(out_text), file=out_file_handle)


def ctm_line_to_string(ctm_line):
    line = list(ctm_line[0:5])

    if len(ctm_line) == 5:
        line.append(1.0)
    elif len(ctm_line) == 6:
        line.append(ctm_line[5])
    else:
        if len(ctm_line) != 8:
            raise RuntimeError("len(ctm_line) expected to be {0}. "
                               "Invalid line {1}".format(8, ctm_line))
        line.append(ctm_line[6])
        line.append(ctm_line[7])

    return " ".join([str(x) for x in line])


def _run(args):
    if args.debug_only:
        test_alignment()
        raise SystemExit("Exiting since --debug-only was true")

    similarity_score = (lambda x, y: args.correct_score
                                     if x == y
                                     else -args.substitution_penalty)

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

    if args.hyp_format == "Text":
        hyp_lines = {key: value
                     for (key, value) in read_text(args.hyp_in_file)}
    else:
        hyp_lines = read_ctm(args.hyp_in_file, file_and_channel2reco)

    num_err = 0
    for reco, ref_text in read_text(args.ref_in_file):
        if reco not in hyp_lines:
            num_err += 1
            raise Warning("Could not find recording {0} "
                          "in hypothesis {1}".format(reco,
                                                     args.hyp_in_file.name))
            continue

        hyp = hyp_lines[reco]

        if args.reco2file_and_channel is not None:
            file_, channel = reco2file_and_channel[reco]
        else:
            file_ = reco
            channel = "1"
        [output, score] = smith_waterman_alignment(
            ref_text, hyp, args.eps_symbol,
            similarity_score, del_score, ins_score,
            hyp_format=args.hyp_format, file_=file_, channel=channel)

        print_alignment(reco, reco2file_and_channel, output,
                        args.eps_symbol, args.hyp_format,
                        args.alignment_out_file)


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
