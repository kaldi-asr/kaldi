#! /usr/bin/env python

"""This script finds retrieves documents similar to the query documents
using a similarity score based on the total TFIDF for all the terms in the
query document.
"""

from __future__ import print_function
import argparse
import logging
import sys

import tf_idf
sys.path.insert(0, 'steps')
import libs.exceptions as kaldi_exceptions


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

    parser.add_argument("--verbosity", type=int, default=0,
                        help="Higher for more logging statements")
    parser.add_argument("--source-text2docs", type=argparse.FileType('r'),
                        required=True,
                        help="""A mapping from the source text to a list of
                        documents that it is broken into
                        <text-utterance-id> <document-id-1> ...
                        <document-id-N>""")
    parser.add_argument("--query-doc2source", type=argparse.FileType('r'),
                        required=True,
                        help="""A mapping from the query document-id to a
                        source text from which a document needs to be
                        retrieved.""")
    parser.add_argument("--num-neighbors-to-search", type=int, default=0,
                        help="""Number of neighboring documents to search
                        around the one retrieved based on maximum tf-idf
                        similarity.""")
    parser.add_argument("--neighbor-tfidf-threshold", type=float, default=0,
                        help="""Ignore neighbors that have tf-idf similarity
                        with the query document less than this threshold
                        lower than the best best score.""")
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
        parts = line.strip().split()
        key = parts[0]

        if (num_values_per_key is not None
                and len(parts) - 1 != num_values_per_key):
            raise kaldi_exceptions.InputError(
                "Expecting {0} columns; Got {1}.".format(
                    num_values_per_key + 1, len(parts)),
                line=line, input_file=file_handle.name)

        if (min_num_values_per_key is not None
                and len(parts) - 1 < min_num_values_per_key):
            raise kaldi_exceptions.InputError(
                "Expecting at least {0} columns; Got {1}.".format(
                    min_num_values_per_key + 1, len(parts)),
                line=line, input_file=file_handle.name)

        if must_contain_unique_key and key in dict_map:
            raise kaldi_exceptions.KeyNotUniqueError(
                key=key, location=file_handle.name)

        if num_values_per_key is not None and num_values_per_key == 1:
            dict_map[key] = parts[1]
        else:
            dict_map[key] = parts[1:]
    file_handle.close()
    return dict_map


def _run(args):
    """The main function that does all the processing.
    Takes as argument the Namespace object obtained from _get_args().
    """
    query_doc2source = read_map(args.query_doc2source, num_values_per_key=1)
    source_text2docs = read_map(args.source_text2docs,
                                min_num_values_per_key=1)

    source_tfidf = tf_idf.TFIDF()
    source_tfidf.read(args.source_tfidf)

    num_queries = 0
    for query_id, query_tfidf in tf_idf.read_tfidf_ark(args.query_tfidf):
        num_queries += 1

        # The source text from which a document is to be retrieved for the
        # input query
        source_text = query_doc2source[query_id]

        # The source documents corresponding to the source text.
        # This is set of documents which will be searched over for the query.
        source_docs = source_text2docs[source_text]

        scores = query_tfidf.compute_similarity_scores(
            source_tfidf, source_docs=source_docs, query_id=query_id)

        assert len(scores) > 0, (
            "Did not get scores for query {0}".format(query_id))

        if args.verbosity > 4:
            for tup, score in scores.iteritems():
                logger.debug("Score, {num}: {0} {1} {2}".format(
                    tup[0], tup[1], score, num=num_queries))

        best_index, best_doc = max(
            enumerate(source_docs), key=lambda x: scores[(query_id, x[1])])
        best_score = scores[(query_id, best_doc)]

        assert source_docs[best_index] == best_doc
        assert best_score == max([scores[(query_id, x)] for x in source_docs])

        best_docs = [best_doc]
        if args.num_neighbors_to_search > 0:
            # Include indexes neighboring to the best document
            indexes = range(max(best_index - args.num_neighbors_to_search, 0),
                            min(best_index + args.num_neighbors_to_search + 1,
                                len(source_docs)))

            assert len(indexes) > 0

            best_docs = [source_docs[i] for i in indexes
                         if (scores[(query_id, source_docs[i])]
                             >= best_score - args.neighbor_tfidf_threshold)]

            assert len(best_docs) > 0, (
                "Did not get best docs for query {0}\n"
                "Scores: {1}\n"
                "Source docs: {2}\n"
                "Best index: {best_index}, score: {best_score}\n".format(
                    query_id, scores, source_docs,
                    best_index=best_index, best_score=best_score))
        assert best_doc in best_docs

        print ("{0} {1}".format(query_id, " ".join(best_docs)),
               file=args.relevant_docs)
    logger.info("Retrieved similar documents for "
                "{0} queries".format(num_queries))


def main():
    args = _get_args()

    if args.verbosity > 1:
        handler.setLevel(logging.DEBUG)
    try:
        _run(args)
    finally:
        args.query_doc2source.close()
        args.source_text2docs.close()
        args.relevant_docs.close()
        args.query_tfidf.close()
        args.source_tfidf.close()


if __name__ == '__main__':
    main()
