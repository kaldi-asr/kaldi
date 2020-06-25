#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This script reads an archive of mapping from query to
documents and stitches the documents for each query into a
new document.
Here "document" is just a list of words.

query2docs is a mapping from query-id to a list of tuples
(document-id, start-fraction, end-fraction)
The tuple can be just the document-id, which is equivaluent to
specifying a start-fraction and end-fraction of 1.0
The start and end fractions are used to stitch only a part of the
document to the retrieved set for the query.

e.g.
query1 doc1 doc2
query2 doc1,0,0.3 doc2,1,1

input-documents
doc1 A B C
doc2 D E
output-documents
query1 A B C D E
query2 C D E
"""

from __future__ import print_function
import argparse
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)

for l in [logger, logging.getLogger('libs')]:
    l.setLevel(logging.DEBUG)
    l.addHandler(handler)


def get_args():
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
                        choices=["true", "false"], default="false",
                        help="If specified, the script will expect "
                        "the document ids in --query2docs to be "
                        "sorted.")

    args = parser.parse_args()

    args.check_sorted_docs_per_query = bool(
        args.check_sorted_docs_per_query == "true")

    return args


def run(args):
    documents = {}
    for line in args.input_documents:
        parts = line.strip().split()
        key = parts[0]
        documents[key] = parts[1:]
    args.input_documents.close()

    for line in args.query2docs:
        try:
            parts = line.strip().split()
            query = parts[0]
            document_infos = parts[1:]

            output_document = []
            prev_doc_id = ''
            for doc_info in document_infos:
                try:
                    doc_id, start_fraction, end_fraction = doc_info.split(',')
                    start_fraction = float(start_fraction)
                    end_fraction = float(end_fraction)
                except ValueError:
                    doc_id = doc_info
                    start_fraction = 1.0
                    end_fraction = 1.0

                if args.check_sorted_docs_per_query:
                    if prev_doc_id != '':
                        if doc_id <= prev_doc_id:
                            raise RuntimeError(
                                "Documents not sorted and "
                                "--check-sorted-docs-per-query was True; "
                                "{0} <= {1}".format(doc_id, prev_doc_id))
                    prev_doc_id = doc_id

                doc = documents[doc_id]
                num_words = len(doc)

                if start_fraction == 1.0 or end_fraction == 1.0:
                    assert end_fraction == end_fraction
                    output_document.extend(doc)
                else:
                    assert (start_fraction + end_fraction < 1.0)
                    if start_fraction > 0:
                        output_document.extend(
                            doc[0:int(start_fraction * num_words)])
                    if end_fraction > 0:
                        output_document.extend(
                            doc[int(end_fraction * num_words):])

            print ("{0} {1}".format(query, " ".join(output_document)),
                   file=args.output_documents)
        except Exception:
            logger.error("Error processing line %s in file %s", line,
                         args.query2docs.name)
            raise


def main():
    args = get_args()

    try:
        run(args)
    except:
        logger.error("Failed to stictch document; got error ",
                     exc_info=True)
        raise SystemExit(1)
    finally:
        for f in [args.query2docs, args.input_documents,
                  args.output_documents]:
            if f is not None:
                f.close()


if __name__ == '__main__':
    main()
