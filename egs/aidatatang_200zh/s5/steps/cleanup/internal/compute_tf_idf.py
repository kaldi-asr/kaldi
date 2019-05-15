#! /usr/bin/env python

from __future__ import print_function
import argparse
import logging
import sys

import tf_idf
sys.path.insert(0, 'steps')

logger = logging.getLogger('tf_idf')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _get_args():
    parser = argparse.ArgumentParser(
        description="""This script takes in a set of documents and computes the
        TF-IDF for each n-gram up to the specified order.  The script can also
        load IDF stats from a different file instead of computing them from the
        input set of documents.""")

    parser.add_argument("--tf-weighting-scheme", type=str, default="raw",
                        choices=["binary", "raw", "log", "normalized"],
                        help="""The function applied on the raw
                        term-frequencies f(t,d) when computing tf(t,d).
                        TF weighting schemes:-
                        binary : tf(t,d) = 1 if t in d else 0
                        raw    : tf(t,d) = f(t,d)
                        log    : tf(t,d) = 1 + log(f(t,d))
                        normalized : tf(t,d) = K + (1-K) * """
                        """f(t,d) / max{f(t',d): t' in d}""")
    parser.add_argument("--tf-normalization-factor", type=float, default=0.5,
                        help="K value for normalized TF weighting scheme")
    parser.add_argument("--idf-weighting-scheme", type=str, default="log",
                        choices=["unary", "log", "log-smoothed",
                                 "probabilistic"],
                        help="""The function applied on the raw
                        inverse-document frequencies n(t) = |d in D: t in d|
                        when computing idf(t,d).
                        IDF weighting schemes:-
                        unary  : idf(t,D) = 1
                        log    : idf(t,D) = log (N / 1 + n(t))
                        log-smoothed : idf(t,D) = log(1 + N / n(t))
                        probabilistic: idf(t,D) = log((N - n(t)) / n(t))""")
    parser.add_argument("--ngram-order", type=int, default=2,
                        help="Accumulate for terms upto this n-grams order")

    parser.add_argument("--input-idf-stats", type=argparse.FileType('r'),
                        help="If provided, IDF stats are loaded from this "
                        "file")
    parser.add_argument("--output-idf-stats", type=argparse.FileType('w'),
                        help="If providied, IDF stats are written to this "
                        "file")
    parser.add_argument("--accumulate-over-docs", type=str, default="true",
                        choices=["true", "false"],
                        help="If true, the stats are accumulated over all the "
                        "documents and a single tf-idf-file is written out.")
    parser.add_argument("docs", type=argparse.FileType('r'),
                        help="Input documents in kaldi text format i.e. "
                        "<document-id> <text>")
    parser.add_argument("tf_idf_file", type=argparse.FileType('w'),
                        help="Output tf-idf for each (t,d) pair in the "
                        "input documents written in the format "
                        "<terms> <document-id> <tf-idf>")

    args = parser.parse_args()

    if args.tf_normalization_factor >= 1.0 or args.tf_normalization_factor < 0:
        raise ValueError("--tf-normalization-factor must be in [0,1)")

    args.accumulate_over_docs = bool(args.accumulate_over_docs == "true")

    if not args.accumulate_over_docs and args.input_idf_stats is None:
        raise TypeError(
            "If --accumulate-over-docs=false is provided, "
            "then --input-idf-stats must be provided.")

    return args


def _run(args):
    tf_stats = tf_idf.TFStats()
    idf_stats = tf_idf.IDFStats()

    if args.input_idf_stats is not None:
        idf_stats.read(args.input_idf_stats)

    num_done = 0
    for line in args.docs:
        parts = line.strip().split()
        doc = parts[0]
        tf_stats.accumulate(doc, parts[1:], args.ngram_order)

        if not args.accumulate_over_docs:
            # Write the document-id and the corresponding tf-idf values.
            print (doc, file=args.tf_idf_file, end=' ')
            tf_idf.write_tfidf_from_stats(
                tf_stats, idf_stats, args.tf_idf_file,
                tf_weighting_scheme=args.tf_weighting_scheme,
                idf_weighting_scheme=args.idf_weighting_scheme,
                tf_normalization_factor=args.tf_normalization_factor,
                expected_document_id=doc)
            tf_stats = tf_idf.TFStats()
        num_done += 1

    if args.accumulate_over_docs:
        tf_stats.compute_term_stats(idf_stats=idf_stats
                                              if args.input_idf_stats is None
                                              else None)

        if args.output_idf_stats is not None:
            idf_stats.write(args.output_idf_stats)
            args.output_idf_stats.close()

        tf_idf.write_tfidf_from_stats(
            tf_stats, idf_stats, args.tf_idf_file,
            tf_weighting_scheme=args.tf_weighting_scheme,
            idf_weighting_scheme=args.idf_weighting_scheme,
            tf_normalization_factor=args.tf_normalization_factor)

    if num_done == 0:
        raise RuntimeError("Could not compute TF-IDF for any query documents")

def main():
    args = _get_args()

    try:
        _run(args)
    finally:
        if args.input_idf_stats is not None:
            args.input_idf_stats.close()
        if args.output_idf_stats is not None:
            args.output_idf_stats.close()
        args.docs.close()
        args.tf_idf_file.close()


if __name__ == '__main__':
    main()
