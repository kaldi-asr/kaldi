#! /usr/bin/env python

from __future__ import print_function
import sys, argparse, imp

verbosity = 0

def GetArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hyp-format", type=str, choices = ["Text", "CTM"],
                        "Format used for the hypothesis")
    parser.add_argument("--eps-symbol", type=str, default = "-",
                        "Symbol used to contain alignment to empty symbol")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--correct-score", type=int, default=1,
                        "Score for correct matches")
    parser.add_argument("--substitution-penalty", type=int, default=1,
                        "Penalty for substitution errors")
    parser.add_argument("--deletion-penalty-type", type=str, default = "linear",
                        choices = ["linear"],
                        help = "Type of function for deletion penalty")
    parser.add_argument("--deletion-penalty-scale", type-int, default=1,
                        "Scale on deletion penalty")
    parser.add_argument("--insertion-penalty-type", type=str, default = "linear",
                        choices = ["linear"],
                        help = "Type of function for insertion penalty")
    parser.add_argument("--insertion-penalty-scale", type-int, default=1,
                        "Scale on insertion penalty")
    parser.add_argument("--debug-only", type=str, default = "false",
                        choices = ["true", "false"],
                        help = "Run test functions only")
    parser.add_argument("ref", type=str,
                        help = "Reference text")
    parser.add_argument("hyp", type=str,
                        help = "Hypothesis text or CTM")

    args = parser.parse_args()
    verbosity = args.verbose
    args.debug_only = True if args.debug_only == "true" else False

    return args

def SmithWatermanAlignment(ref, hyp, eps_symbol,
                           SimilarityScore, DelScore, InsScore,
                           hyp_format = "CTM"):
    output = []

    M = len(ref)
    N = len(hyp)

    bp = [ [ ] for x in range(M+1) ]
    H = [ [ ] for x in range(M+1) ]

    for m in range(M+1):
        H[m] = [ 0 for x in range(N+1) ]
        bp[m] = [ (0,0) for x in range(N+1) ]

    global_max = 0
    global_max_element = (0,0)

    for m in range(1, M+1):
        for n in range(1, N+1):
            sub_or_ok = (H[m-1][n-1] +
                         SimilarityScore(ref[m-1],
                                         hyp[n-1][3] if hyp_format == "CTM"
                                                   else hyp[n-1]))
            if sub_or_ok > 0:
                H[m][n] = sub_or_ok
                bp[m][n] = (m-1, n-1)
                if verbosity > 2:
                    print ("({0},{1}) -> ({2},{3}): {4} ({5},{6})".format(m-1,n-1,m,n, H[m][n], ref[m-1], hyp[n-1]), file = sys.stderr)

            for k in range(1, m+1):
                if H[m-k][n] + DelScore(k) > H[m][n]:
                    H[m][n] = H[m-k][n] + DelScore(k)
                    bp[m][n] = (m-k, n)
                    if verbosity > 2:
                        print ("({0},{1}) -> ({2},{3}): {4}".format(m-k,n,m,n, H[m][n]), file = sys.stderr)

            for l in range(1, n+1):
                if H[m][n-l] + InsScore(l) > H[m][n]:
                    H[m][n] = H[m][n-l] + InsScore(l)
                    bp[m][n] = (m, n-l)
                    if verbosity > 2:
                        print ("({0},{1}) -> ({2},{3}): {4}".format(m,n-l,m,n, H[m][n]), file = sys.stderr)

            if H[m][n] > global_max:
                global_max = H[m][n]
                global_max_element = (m,n)

    m,n = global_max_element
    score = global_max

    #output.append( (m, n,
    #                ref[m-1] if m > 0 else eps_symbol,
    #                hyp[n-1] if n > 0 else eps_symbol) )

    while score > 0:
        prev_m, prev_n = bp[m][n]

        if (m == prev_m + 1 and n == prev_n + 1):
            # Substitution or correct
            output.append((ref[m-1] if m > 0 else eps_symbol,
                           hyp[n-1] if n > 0 else eps_symbol,
                           prev_m, prev_n, m, n))
        elif (prev_n == n):
            # Deletion
            output.append((ref[m-1] if m > 0 else eps_symbol,
                           eps_symbol,
                           prev_m, prev_n, m, n))
        elif (prev_m == m):
            # Insertion
            output.append((eps_symbol,
                           hyp[n-1] if n > 0 else eps_symbol,
                           prev_m, prev_n, m, n))
        else:
            if (prev_m, prev_n) != 0:
                raise Exception("Unexpected result: Bug in code!!")

        m,n = (prev_m, prev_n)
        score = H[m][n]

    assert(score == 0)

    output.reverse()

    if verbosity > 1:
        for m in range(M+1):
            for n in range(N+1):
                sys.stderr.write("{0} ".format(H[m][n]))
            print ("", file = sys.stderr)


        print ("\t-\t".join("({0},{1})".format(x[4],x[5]) for x in output), file = sys.stderr)
        print ("\t\t".join(str(x[0]) for x in output), file = sys.stderr)
        print ("\t\t".join(str(x[1]) for x in output), file = sys.stderr)
        print (str(score), file = sys.stderr)

    return (output, global_max)

def TestAlignment():
    hyp = "ACACACTA"
    ref = "AGCACACA"

    output,score = SmithWatermanAlignment(ref, hyp, "-", lambda x,y: 2 if (x == y) else -1,
                                          lambda x: -x, lambda x: -x, hyp_format = "Text")

def ReadText(text_file):
    text_lines = {}
    for line in open(text_file).readlines():
        parts = line.strip().split()
        text_lines[parts[0]] = parts[1]
    return text_lines

def ReadCtm(ctm_file):
    all_ctm = {}
    prev_reco = ""
    ctm_lines = []
    for line in open(ctm_file).readlines():
        parts = line.strip().split()
        reco = parts[0]
        if prev_reco != "" and reco != prev_reco:
            # New recording
            all_ctm[prev_reco] = ctm_lines
            ctm_lines = []
        ctm_lines.append(parts)
        prev_reco = reco
    return all_ctm

def Main():
    args = GetArgs()

    if args.debug_only:
        TestAlignment()
        sys.exit(0)

    SimilarityScore = (lambda x,y: args.correct_score if (x == y)
                                   else -args.substitution_penalty)

    if args.deletion_penalty_type == "linear":
        DelScore = lambda x: -args.deletion_penalty_scale * x
    else:
        raise NotImplementedError("Only linear deletion-penalty-type is implemented")

    if args.insertion_penalty_type == "linear":
        InsScore = lambda x: -args.insertion_penalty_scale * x
    else:
        raise NotImplementedError("Only linear insertion-penalty-type is implemented")

    ref_lines = ReadText(args.ref)

    if args.hyp_format == "Text":
        hyp_lines = ReadText(args.hyp)
    else:
        hyp_lines = ReadCtm(args.hyp)

    num_err = 0
    for reco, ref_text in ref_lines.iteritems():
        if reco not in hyp_lines:
            num_err += 1
            raise Warning("Could not find recording {0} in hypothesis {1}".format(reco, args.hyp))
            continue

        hyp = hyp_lines[reco]
        [output, score] = SmithWatermanAlignment(ref_text, hyp, args.eps_symbol,
                                                 SimilarityScore, DelScore, InsScore,
                                                 hyp_format = args.hyp_format)

        PrintAlignment(output)

if __name__ == '__main__':
    Main()
