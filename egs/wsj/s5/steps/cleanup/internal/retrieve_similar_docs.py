#! /usr/bin/env python

"""This script finds retrieves documents similar to the query documents
using a similarity score based on the total TFIDF for all the terms in the
query document.
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


def _get_args():
    parser = argparse.ArgumentParser(
        description="""This script finds retrieves documents similar to the
        query documents using a similarity score based on the total TFIDF for
        all the terms in the query document.""")

    parser.add_argument("--partial-doc-fraction", default=0.2,
                        help="""The fraction of neighboring document that will
                        be part of the retrieved document set.""")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Higher for more logging statements")
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
    parser.add_argument("--num-neighbors-to-search", type=int, default=0,
                        help="""Number of neighboring documents to search
                        around the one retrieved based on maximum tf-idf
                        similarity.""")
    parser.add_argument("--neighbor-tfidf-threshold", type=float, default=0.9,
                        help="""Ignore neighbors that have tf-idf similarity
                        with the query document less than this threshold
                        factor lower than the best best score.""")
    parser.add_argument("--query-tfidf", type=argparse.FileType('r'),
                        required=True,
                        help="""Archive of TF-IDF for query documents
                        indexed by the document-id""")
    parser.add_argument("--source-tfidf", type=argparse.FileType('r'),
                        required=True,
                        help="""TF-IDF for source documents that need to be
                        retrieved""")
    parser.add_argument("--relevant-docs", type=argparse.FileType('w'),
                        required=True,
                        help="""Archive of a list of source documents
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
        num_values_per_key - If provided, the function raises an InputError if
                             the number of values read for a key in the input
                             file does not match the "num_values_per_key"
        min_num_values_per_key - If provided, the function raises an InputError
                                 if the number of values read for a key in the
                                 input file is less than
                                 "min_num_values_per_key"
        must_contain_unique_key - If provided, the function raises a
                                  KeyNotUniqueError when a duplicate key is
                                  read from the file.

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
        [(key, value[0], value[1]) for key, value in indexes.iteritems()],
        key=lambda x: x[0])

    doc_ids = []
    for i, partial_start, partial_end in indexes:
        try:
            doc_ids.append((source_docs[i], partial_start, partial_end))
        except IndexError:
            pass
    return doc_ids


def _run(args):
    """The main function that does all the processing.
    Takes as argument the Namespace object obtained from _get_args().
    """
    query_id2source_text_id = read_map(args.query_id2source_text_id,
                                       num_values_per_key=1)
    source_text_id2doc_ids = read_map(args.source_text_id2doc_ids,
                                      min_num_values_per_key=1)

    source_tfidf = tf_idf.TFIDF()
    source_tfidf.read(args.source_tfidf)

    num_queries = 0
    for query_id, query_tfidf in tf_idf.read_tfidf_ark(args.query_tfidf):
        num_queries += 1

        # The source text from which a document is to be retrieved for the
        # input query
        source_text_id = query_id2source_text_id[query_id]

        # The source documents corresponding to the source text.
        # This is set of documents which will be searched over for the query.
        source_doc_ids = source_text_id2doc_ids[source_text_id]

        scores = query_tfidf.compute_similarity_scores(
            source_tfidf, source_docs=source_doc_ids, query_id=query_id)

        assert len(scores) > 0, (
            "Did not get scores for query {0}".format(query_id))

        if args.verbose > 2:
            for tup, score in scores.iteritems():
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
    logger.info("Retrieved similar documents for "
                "%d queries", num_queries)


def main():
    args = _get_args()

    if args.verbose > 1:
        handler.setLevel(logging.DEBUG)
    try:
        _run(args)
    finally:
        args.query_id2source_text_id.close()
        args.source_text_id2doc_ids.close()
        args.relevant_docs.close()
        args.query_tfidf.close()
        args.source_tfidf.close()


if __name__ == '__main__':
    main()
