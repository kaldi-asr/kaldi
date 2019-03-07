#! /usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0.

"""This script retrieves documents similar to the query documents
using a similarity score based on the total TFIDF for all the terms in the
query document.

Some terminology:
    original utterance-id = The utterance-id of the original long audio segments
        and the corresponding reference transcript
    source-text = reference transcript
    source-text-id = original utterance-id
    sub-segment = Approximately 30s long chunk of the original utterance
    query-id = utterance-id of the sub-segment
    document = Approximately 1000 words of a source-text
    doc-id = Id of the document

e.g.
foo1 A B C D E F is in the original text file
and foo1 foo 100 200 is in the original segments file.

Here foo1 is the source-text-id and "A B C D" is the reference transcript. It
is a 100s long segment from the recording foo.

foo1 is split into 30s long sub-segments as follows:
foo1-1 foo1 100 130
foo1-2 foo1 125 155
foo1-3 foo1 150 180
foo1-4 foo1 175 200

foo1-{1,2,3,4} are query-ids.

The source-text for foo1 is split into two-word documents.
doc1 A B
doc2 C D
doc3 E F

doc{1,2,3} are doc-ids.

--source-text2doc-ids option is given a mapping that contains
foo1 doc1 doc2 doc3

--query-id2source-text-id option is given a mapping that contains
foo1-1 foo1
foo1-2 foo1
foo1-3 foo1
foo1-4 foo1

The query TF-IDFs are all indexed by the utterance-id of the sub-segments
of the original utterances.
The source TF-IDFs use the document-ids created by splitting the source-text
(corresponding to original utterances) into documents.

For each query (sub-segment), we need to retrieve the documents that were
created from the same original utterance that the sub-segment was from. For
this, we have to load the source TF-IDF that has those documents. This
information is provided using the option --source-text2tf-idf-file, which
is like an SCP file with the first column being the source-text-id and the
second column begin the location of TF-IDF for the documents corresponding
to that source-text-id.

The output of this script is a file where the first column is the
query-id (i.e. sub-segment-id) and the remaining columns, which is at least
one in number and a maxmium of (1 + 2 * num-neighbors-to-search) columns
are tuples separated by commas
(<doc-id>, <start-fraction>, <end-fraction>), where <doc-id> is the document-id
<start-fraction> is the proportion of the document from the beginning
that needs to be in the retrieved set.
<end-fraction> is the proportion of the document from the end
that needs to be in the retrieved set.
If both <start-fraction> and <end-fraction> are 1, then the full document is
added to the retrieved set.
Some examples of the lines in the output file are:
foo1-1 doc1,1,1
foo1-2 doc1,0,0.2 doc2,1,1 doc3,0.2,0
"""

from __future__ import print_function
import argparse
import logging

import tf_idf


logger = logging.getLogger('__name__')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)

for l in [logger, logging.getLogger('tf_idf'), logging.getLogger('libs')]:
    l.setLevel(logging.DEBUG)
    l.addHandler(handler)


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script retrieves documents similar to the
        query documents using a similarity score based on the total TFIDF for
        all the terms in the query document.
        See the beginning of the script for more details about the
        arguments to the script.""")

    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Higher for more logging statements")

    parser.add_argument("--num-neighbors-to-search", type=int, default=0,
                        help="""Number of neighboring documents to search
                        around the one retrieved based on maximum tf-idf
                        similarity. A value of 0 means only the document
                        with the maximum tf-idf similarity is retrieved,
                        and none of the documents adjacent to it.""")
    parser.add_argument("--neighbor-tfidf-threshold", type=float, default=0.9,
                        help="""Ignore neighbors that have tf-idf similarity
                        with the query document less than this threshold
                        factor lower than the best score.""")
    parser.add_argument("--partial-doc-fraction", default=0.2,
                        help="""The fraction of neighboring document that will
                        be part of the retrieved document set.
                        If this is greater than 0, then a fraction of words
                        from the neighboring documents is added to the
                        retrieved document.""")

    parser.add_argument("--source-text-id2doc-ids",
                        type=argparse.FileType('r'), required=True,
                        help="""A mapping from the source text to a list of
                        documents that it is broken into
                        <text-utterance-id> <document-id-1> ...
                        <document-id-N>""")
    parser.add_argument("--query-id2source-text-id",
                        type=argparse.FileType('r'), required=True,
                        help="""A mapping from the query document-id to a
                        source text from which a document needs to be
                        retrieved.""")
    parser.add_argument("--source-text-id2tfidf", type=argparse.FileType('r'),
                        required=True,
                        help="""An SCP file for the TF-IDF for source
                        documents indexed by the source-text-id.""")
    parser.add_argument("--query-tfidf", type=argparse.FileType('r'),
                        required=True,
                        help="""Archive of TF-IDF objects for query documents
                        indexed by the query-id.
                        The format is
                        query-id <TFIDF> ... </TFIDF>
                        """)
    parser.add_argument("--relevant-docs", type=argparse.FileType('w'),
                        required=True,
                        help="""Output archive of a list of source documents
                        similar to a query document, indexed by the
                        query document id.""")

    args = parser.parse_args()

    if args.partial_doc_fraction < 0 or args.partial_doc_fraction > 1:
        logger.error("--partial-doc-fraction must be in [0,1]")
        raise ValueError

    return args


def read_map(file_handle, num_values_per_key=None,
             min_num_values_per_key=None, must_contain_unique_key=True):
    """Reads a map from a file into a dictionary and returns it.
    Expects the map is stored in the file in the following format:
    <key> <value-1> <value-2> ... <value-N>
    The values are returned as a tuple stored in a dictionary indexed by the
    "key".

    Arguments:
        file_handle - A handle to an opened input file containing the map
        num_values_per_key - If provided, the function raises an error if
                             the number of values read for a key in the input
                             file does not match the "num_values_per_key"
        min_num_values_per_key - If provided, the function raises an error
                                 if the number of values read for a key in the
                                 input file is less than
                                 "min_num_values_per_key"
        must_contain_unique_key - If set to True, then it is required that the
                                  file has a unique key; otherwise this
                                  function will exit with error.

    Returns:
        { key: tuple(values) }
    """
    dict_map = {}
    for line in file_handle:
        try:
            parts = line.strip().split()
            key = parts[0]

            if (num_values_per_key is not None
                    and len(parts) - 1 != num_values_per_key):
                logger.error(
                    "Expecting {0} columns; Got {1}.".format(
                        num_values_per_key + 1, len(parts)))
                raise TypeError

            if (min_num_values_per_key is not None
                    and len(parts) - 1 < min_num_values_per_key):
                logger.error(
                    "Expecting at least {0} columns; Got {1}.".format(
                        min_num_values_per_key + 1, len(parts)))
                raise TypeError

            if must_contain_unique_key and key in dict_map:
                logger.error("Found duplicate key %s", key)
                raise TypeError

            if num_values_per_key is not None and num_values_per_key == 1:
                dict_map[key] = parts[1]
            else:
                dict_map[key] = parts[1:]
        except Exception:
            logger.error("Failed reading line %s in file %s",
                         line, file_handle.name)
            raise
    file_handle.close()
    return dict_map


def get_document_ids(source_docs, indexes):
    indexes = sorted(
        [(key, value[0], value[1]) for key, value in indexes.items()],
        key=lambda x: x[0])

    doc_ids = []
    for i, partial_start, partial_end in indexes:
        try:
            doc_ids.append((source_docs[i], partial_start, partial_end))
        except IndexError:
            pass
    return doc_ids


def run(args):
    """The main function that does all the processing.
    Takes as argument the Namespace object obtained from _get_args().
    """
    query_id2source_text_id = read_map(args.query_id2source_text_id,
                                       num_values_per_key=1)
    source_text_id2doc_ids = read_map(args.source_text_id2doc_ids,
                                      min_num_values_per_key=1)

    source_text_id2tfidf = read_map(args.source_text_id2tfidf,
                                    num_values_per_key=1)

    num_queries = 0
    prev_source_text_id = ""
    for query_id, query_tfidf in tf_idf.read_tfidf_ark(args.query_tfidf):
        num_queries += 1

        # The source text from which a document is to be retrieved for the
        # input query
        source_text_id = query_id2source_text_id[query_id]

        if prev_source_text_id != source_text_id:
            source_tfidf = tf_idf.TFIDF()
            source_tfidf.read(
                open(source_text_id2tfidf[source_text_id]))
            prev_source_text_id = source_text_id

        # The source documents corresponding to the source text.
        # This is set of documents which will be searched over for the query.
        source_doc_ids = source_text_id2doc_ids[source_text_id]

        scores = query_tfidf.compute_similarity_scores(
            source_tfidf, source_docs=source_doc_ids, query_id=query_id)

        assert len(scores) > 0, (
            "Did not get scores for query {0}".format(query_id))

        if args.verbose > 2:
            for tup, score in scores.items():
                logger.debug("Score, {num}: {0} {1} {2}".format(
                    tup[0], tup[1], score, num=num_queries))

        best_index, best_doc_id = max(
            enumerate(source_doc_ids), key=lambda x: scores[(query_id, x[1])])
        best_score = scores[(query_id, best_doc_id)]

        assert source_doc_ids[best_index] == best_doc_id
        assert best_score == max([scores[(query_id, x)]
                                  for x in source_doc_ids])

        best_indexes = {}

        if args.num_neighbors_to_search == 0:
            best_indexes[best_index] = (1, 1)
            if best_index > 0:
                best_indexes[best_index - 1] = (0, args.partial_doc_fraction)
            if best_index < len(source_doc_ids) - 1:
                best_indexes[best_index + 1] = (args.partial_doc_fraction, 0)
        else:
            excluded_indexes = set()
            for index in range(
                    max(best_index - args.num_neighbors_to_search, 0),
                    min(best_index + args.num_neighbors_to_search + 1,
                        len(source_doc_ids))):
                if (scores[(query_id, source_doc_ids[index])]
                        >= args.neighbor_tfidf_threshold * best_score):
                    best_indexes[index] = (1, 1)    # Type 2
                    if index > 0 and index - 1 in excluded_indexes:
                        try:
                            # Type 1 and 3
                            start_frac, end_frac = best_indexes[index - 1]
                            assert end_frac == 0
                            best_indexes[index - 1] = (
                                start_frac, args.partial_doc_fraction)
                        except KeyError:
                            # Type 1
                            best_indexes[index - 1] = (
                                0, args.partial_doc_fraction)
                else:
                    excluded_indexes.add(index)
                    if index > 0 and index - 1 not in excluded_indexes:
                        # Type 3
                        best_indexes[index] = (args.partial_doc_fraction, 0)

        best_docs = get_document_ids(source_doc_ids, best_indexes)

        assert len(best_docs) > 0, (
            "Did not get best docs for query {0}\n"
            "Scores: {1}\n"
            "Source docs: {2}\n"
            "Best index: {best_index}, score: {best_score}\n".format(
                query_id, scores, source_doc_ids,
                best_index=best_index, best_score=best_score))
        assert (best_doc_id, 1.0, 1.0) in best_docs

        print ("{0} {1}".format(query_id, " ".join(
            ["%s,%.2f,%.2f" % x for x in best_docs])),
               file=args.relevant_docs)

    if num_queries == 0:
        raise RuntimeError("Failed to retrieve any document.")

    logger.info("Retrieved similar documents for "
                "%d queries", num_queries)


def main():
    args = get_args()

    if args.verbose > 1:
        handler.setLevel(logging.DEBUG)
    try:
        run(args)
    finally:
        for f in [args.query_id2source_text_id, args.source_text_id2doc_ids,
                  args.relevant_docs, args.query_tfidf, args.source_text_id2tfidf]:
            f.close()


if __name__ == '__main__':
    main()
