#! /usr/bin/python

# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This script reads an archive of mapping from query to
documents and stitches the documents for each query into a
new document.
Here "document" is just a vector of words.
"""

from __future__ import print_function
import argparse
import logging
import sys

sys.path.insert(0, 'steps')
import libs.exceptions as kaldi_exceptions

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)

for l in [logger, logging.getLogger('libs')]:
    l.setLevel(logging.DEBUG)
    l.addHandler(handler)


def _get_args():
    """Returns arguments parsed from command-line."""

    parser = argparse.ArgumentParser(
        description="""This script reads an archive of mapping from query to
        documents and stitches the documents for each query into a new
        document.""")

    parser.add_argument("--query2docs", type=argparse.FileType('r'),
                        required=True,
                        help="""Input file containing an archive
                        of list of documents indexed by a query document
                        id.""")
    parser.add_argument("--input-documents", type=argparse.FileType('r'),
                        required=True,
                        help="""Input file containing the documents
                        indexed by the document id.""")
    parser.add_argument("--output-documents", type=argparse.FileType('w'),
                        required=True,
                        help="""Output documents indexed by the query
                        document-id, obtained by stitching input documents
                        corresponding to the query.""")
    parser.add_argument("--check-sorted-docs-per-query", type=str,
                        choices=["true", "false"], default="true",
                        help="If specified, the script will expect "
                        "the document ids in --query2docs to be "
                        "sorted.")

    args = parser.parse_args()

    args.check_sorted_docs_per_query = bool(
        args.check_sorted_docs_per_query == "true")

    return args


def _run(args):
    documents = {}
    for line in args.input_documents:
        parts = line.strip().split()
        key = parts[0]
        documents[key] = " ".join(parts[1:])
    args.input_documents.close()

    def is_sorted(x, key=lambda x, i: x[i] <= x[i+1]):
        for i in range(len(x) - 1):
            if not key(x, i):
                return False
        return True

    for line in args.query2docs:
        parts = line.strip().split()
        query = parts[0]
        document_ids = parts[1:]

        if args.check_sorted_docs_per_query:
            if not is_sorted(document_ids):
                raise kaldi_exceptions.InputError(
                    "Documents is not sorted for key {0}".format(query),
                    line=line.strip())

        output_document = []
        for doc_id in document_ids:
            output_document.append(documents[doc_id])

        print ("{0} {1}".format(query, " ".join(output_document)),
               file=args.output_documents)


def main():
    args = _get_args()

    try:
        _run(args)
    finally:
        args.query2docs.close()
        args.input_documents.close()
        args.output_documents.close()


if __name__ == '__main__':
    main()
