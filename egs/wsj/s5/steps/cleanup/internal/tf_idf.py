# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This module contains structures to accumulate, store and use stats
for Term-frequency and Inverse-document-frequency values.
"""

from __future__ import print_function
from __future__ import division
import logging
import math
import re
import sys

sys.path.insert(0, 'steps')

logger = logging.getLogger('__name__')
logger.addHandler(logging.NullHandler())


class IDFStats(object):
    """Stores stats for computing inverse-document-frequencies.
    """
    def __init__(self):
        self.num_docs_for_term = {}
        self.num_docs = 0

    def get_inverse_document_frequency(self, term, weighting_scheme="log"):
        """Get IDF for a term.

        Weighting scheme is the function applied on the raw
        inverse-document frequencies n(t) = |d in D: t in d|
        when computing idf(t,d).
        Let N = Total number of documents.

        IDF weighting schemes:-
        unary  : idf(t,D) = 1
        log    : idf(t,D) = log (N / (1 + n(t)))
        log-smoothed : idf(t,D) = log(1 + N / n(t))
        probabilistic: idf(t,D) = log((N - n(t)) / n(t))
        """
        n_t = float(self.num_docs_for_term.get(term, 0))
        num_terms = len(self.num_docs_for_term)

        if num_terms == 0:
            raise RuntimeError("No IDF stats have been accumulated.")

        if weighting_scheme == "unary":
            return 1
        if weighting_scheme == "log":
            return math.log(float(self.num_docs) / (1.0 + n_t))
        if weighting_scheme == "log-smoothed":
            return math.log(1.0 + float(self.num_docs) / (1.0 + n_t))
        if weighting_scheme == "probabilitic":
            return math.log((self.num_docs - n_t - 1) / (1.0 + n_t))

    def accumulate(self, term):
        """Adds one count to the number of docs containing the term "term".
        """
        self.num_docs_for_term[term] = self.num_docs_for_term.get(term, 0) + 1
        if len(term) == 1:
            self.num_docs += 1

    def write(self, file_handle):
        """Writes the IDF stats to file using the format:
        <term-1> <term-2> ... <term-N> <num-docs>
        for n-gram (<term-1>, ... <term-N>)
        """
        for term, num in self.num_docs_for_term.items():
            if num == 0:
                continue
            assert isinstance(term, tuple)
            print ("{term} {n}".format(term=" ".join(term), n=num),
                   file=file_handle)

    def read(self, file_handle):
        """Loads IDF stats from file. """
        for line in file_handle:
            parts = line.strip().split()
            term = tuple(parts[0:-1])
            self.num_docs_for_term[term] = float(parts[-1])
            if len(term) == 1:
                self.num_docs += 1

        if len(self.num_docs_for_term) == 0:
            raise RuntimeError("Read no IDF stats.")


class TFStats(object):
    """Store stats for TF-IDF computation.
    A separate object of IDFStats is stored within this object.
    """
    def __init__(self):
        self.raw_counts = {}
        self.max_counts_for_term = {}

    def get_term_frequency(self, term, doc, weighting_scheme="raw",
                           normalization_factor=0.5):
        """Returns the term-frequency for (term, document) pair.

        The function applied on the raw term-frequencies f(t,d) when computing
        tf(t,d) is specified by the weighting_scheme.
        binary : tf(t,d) = 1 if t in d else 0
        raw    : tf(t,d) = f(t,d)
        log    : tf(t,d) = 1 + log(f(t,d))
        normalized : tf(t,d) = K + (1-K) * f(t,d) / max{f(t',d): t' in d}
        """
        if weighting_scheme == "binary":
            return 1 if (term, doc) in self.raw_counts else 0
        if weighting_scheme == "raw":
            return self.raw_counts.get((term, doc), 0)
        if weighting_scheme == "log":
            if (term, doc) in self.raw_counts:
                return 1 + math.log(self.raw_counts[(term, doc)])
            return 0
        if weighting_scheme == "normalized":
            return (normalization_factor
                    + (1 - normalization_factor)
                    * self.raw_counts.get((term, doc), 0)
                    / (1.0 + self.max_counts_for_term.get(term, 0)))
        raise KeyError("Unknown tf-weighting-scheme {0}".format(
            weighting_scheme))

    def accumulate(self, doc, text, ngram_order):
        """Accumulate raw stats from a document for upto the specified
        ngram-order."""
        for n in range(1, ngram_order + 1):
            for i in range(len(text)):
                term = tuple(text[i:(i+n)])
                self.raw_counts.setdefault((term, doc), 0)
                self.raw_counts[(term, doc)] += 1

    def compute_term_stats(self, idf_stats=None):
        """Compute the maximum counts for each term over all the documents
        based on the stored raw counts."""
        if len(self.raw_counts) == 0:
            raise RuntimeError("No (term, doc) found in tf-stats.")
        for tup, counts in self.raw_counts.items():
            term = tup[0]

            if counts > self.max_counts_for_term.get(term, 0):
                self.max_counts_for_term[term] = counts

            if idf_stats is not None:
                idf_stats.accumulate(term)

    def __str__(self):
        """Returns a string with all the stats in the following format:
        <n-gram order> <term-1> <term-2> ... <term-n> <document-id> <counts>
        """
        lines = []
        for tup, counts in self.raw_counts.items():
            term, doc = tup
            lines.append("{order} {term} {doc} {counts}".format(
                order=len(term), term=" ".join(term),
                doc=doc, counts=counts))
        return "\n".join(lines)

    def read(self, file_handle, ngram_order=None, idf_stats=None):
        """Reads the TF stats stored in a file in the following format:
        <ngram-order> <term-1> <term-2> ... <term-n> <document-id> <counts>

        If idf_stats is provided then idf_stats is accumulated simultaneously.
        """
        for line in file_handle:
            parts = line.strip().split()
            order = parts[0]
            assert len(parts) - 3 == order
            if ngram_order is not None and order > ngram_order:
                continue
            term = tuple(parts[1:(order+1)])
            doc = parts[-2]
            counts = float(parts[-1])

            self.raw_counts[(term, doc)] = counts

            if counts > self.max_counts_for_term.get(term, 0):
                self.max_counts_for_term[term] = counts

            if idf_stats is not None:
                idf_stats.accumulate(term)

        if len(self.raw_counts) == 0:
            raise RuntimeError("Read no TF stats.")


class TFIDF(object):
    """Class to store TF-IDF values for term-document pairs.

    Parameters:
        tf_idf - A dictionary of TF-IDF values indexed by (term, document)
                 tuple as key
    """

    def __init__(self):
        self.tf_idf = {}

    def get_value(self, term, doc):
        """Returns TF-IDF value for (term, doc) tuple if it exists.
        Otherwise returns 0.
        """
        return self.tf_idf[(term, doc)]

    def compute_similarity_scores(self, source_tfidf, source_docs=None,
                                  do_length_normalization=False,
                                  query_id=None):
        """Computes TF-IDF similarity score between each pair of query
        document contained in this object and the source documents
        in the source_tfidf object.

        Arguments:
            source_docs - If provided, the similarity scores are computed
                          for only the source documents contained in
                          source_docs.
            use_average - If True, then the similarity scores is
                          normalized by the length of query. This is usually
                          not required when the scores are only utilized
                          for ranking the source documents.
            query_id - If provided, check that this tf_idf object
                       contains values only for document with id 'query_id'

        Returns a dictionary
            { (query_document_id, source_document_id): similarity_score }
        """
        num_terms_per_doc = {}
        similarity_scores = {}

        for tup, value in self.tf_idf.items():
            term, doc = tup
            num_terms_per_doc[doc] = num_terms_per_doc.get(doc, 0) + 1

            if query_id is not None and doc != query_id:
                raise RuntimeError("TF-IDF contains document {0}, which is "
                                   "not the required query {1}. \n"
                                   "Something wrong in how this TF-IDF object "
                                   "was created or a bug in the "
                                   "calling script.".format(
                                       doc, query_id))

            if source_docs is not None:
                for src_doc in source_docs:
                    try:
                        src_value = source_tfidf.get_value(term, src_doc)
                    except KeyError:
                        logger.debug(
                            "Could not find ({term}, {src}) in "
                            "source_tfidf. "
                            "Choosing a tf-idf value of 0.".format(
                                term=term, src=src_doc))
                        src_value = 0

                    similarity_scores[(doc, src_doc)] = (
                        similarity_scores.get((doc, src_doc), 0)
                        + src_value * value)
            else:
                for src_tup, src_value in source_tfidf.tf_idf.items():
                    similarity_scores[(doc, src_doc)] = (
                        similarity_scores.get((doc, src_doc), 0)
                        + src_value * value)

        if do_length_normalization:
            for doc_pair, value in similarity_scores.items():
                doc, src_doc = doc_pair
                similarity_scores[(doc, src_doc)] = value / num_terms_per_doc[doc]

        if logger.isEnabledFor(logging.DEBUG):
            for doc, count in num_terms_per_doc.items():
                logger.debug(
                    'Seen {0} terms in query document {1}'.format(count, doc))

        return similarity_scores

    def read(self, tf_idf_file):
        """Loads TFIDF object from file."""

        if len(self.tf_idf) != 0:
            raise RuntimeError("TD-IDF object is not empty.")
        seen_footer = False
        line = tf_idf_file.readline()
        parts = line.strip().split()
        if re.search('^<TFIDF>', line) is None:
            raise TypeError(
                "Invalid format of TD-IDF object. "
                "Missing header <TFIDF>; got {0}".format(line))
        assert parts[0] == "<TFIDF>"
        if len(parts) > 1:
            # Read header; go to the rest of line
            line = " ".join(parts[1:])
        else:
            # Nothing in this line. Read the next lines.
            line = tf_idf_file.readline()
        while line:
            parts = line.strip().split()
            if re.search('</TFIDF>', line):
                if len(parts) > 1:
                    raise TypeError(
                        "Expecting footer </TFIDF> "
                        "to be on a separate line; got {0}".format(line))
                assert parts[0] == "</TFIDF>"
                seen_footer = True
                break
            if re.search('<TFIDF>', line):
                raise TypeError("Got unexpected header <TFIDF> in line "
                                "{0}".format(line))

            order = int(parts[0])
            term = tuple(parts[1:(order + 1)])
            doc = parts[-2]
            tfidf = float(parts[-1])

            entry = (term, doc)
            if entry in self.tf_idf:
                raise RuntimeError("Duplicate entry {0} found while reading "
                                   "TFIDF object.".format(entry))
            self.tf_idf[entry] = tfidf

            line = tf_idf_file.readline()
        if not seen_footer:
            raise TypeError(
                "Did not see footer </TFIDF> "
                "in TFIDF object; got {0}".format(line))

        if len(self.tf_idf) == 0:
            raise RuntimeError(
                "Read no TF-IDF values from file {0}".format(tf_idf_file.name))

    def write(self, tf_idf_file):
        """Writes TFIDF object to file."""

        print ("<TFIDF>", file=tf_idf_file)
        for tup, value in self.tf_idf.items():
            term, doc = tup
            print("{order} {term} {doc} {tfidf}".format(
                order=len(term), term=" ".join(term),
                doc=doc, tfidf=value),
                  file=tf_idf_file)
        print ("</TFIDF>", file=tf_idf_file)


def write_tfidf_from_stats(
        tf_stats, idf_stats, tf_idf_file, tf_weighting_scheme="raw",
        idf_weighting_scheme="log", tf_normalization_factor=0.5,
        expected_document_id=None):
    """Writes TF-IDF values to file args.tf_idf_file.
    The format used is
    <ngram-order> <term> <document> <tfidf>.
    Markers "<TFIDF>" and "</TFIDF>" are added for parsing this file
    easily.

    Arguments:
        tf_stats - A TFStats object
        idf_stats - An IDFStats object
        tf_idf_file - Output file to which the TF-IDF values will be written
        tf_weighting_scheme - See doc_string in TFStats class
        idf_weighting_scheme - See doc_string in IDFStats class
        tf_normalization_factor - See doc_string in TFStats class
        document_id - If provided, checks that the TFStats object contains
                      stats only for this document_id.
    """
    if len(tf_stats.raw_counts) == 0:
        raise RuntimeError("Supplied tf-stats object is empty.")

    if idf_stats.num_docs == 0:
        raise RuntimeError("Supplied idf-stats object is empty.")

    print ("<TFIDF>", file=tf_idf_file)
    for tup in tf_stats.raw_counts:
        term, doc = tup

        if expected_document_id is not None and doc != expected_document_id:
            raise RuntimeError("TFStats object contains stats with "
                               "document {0}, "
                               "which is not the specified "
                               "document {1}.".format(doc,
                                                      expected_document_id))

        tf_value = tf_stats.get_term_frequency(
            term, doc,
            weighting_scheme=tf_weighting_scheme,
            normalization_factor=tf_normalization_factor)

        idf_value = idf_stats.get_inverse_document_frequency(
            term, weighting_scheme=idf_weighting_scheme)

        print("{order} {term} {doc} {tfidf}".format(
            order=len(term), term=" ".join(term),
            doc=doc, tfidf=tf_value * idf_value),
              file=tf_idf_file)
    print ("</TFIDF>", file=tf_idf_file)


def read_key(fd):
  """ [str] = read_key(fd)
   Read the utterance-key from the opened ark/stream descriptor 'fd'.
  """
  str = ''
  while 1:
    char = fd.read(1)
    if char == '' : break
    if char == ' ' : break
    str += char
  str = str.strip()
  if str == '': return None # end of file,
  assert(re.match('^[\.a-zA-Z0-9_:-]+$',str) != None) # check format,
  return str


def read_tfidf_ark(file_handle):
    """Read a kaldi archive of TFIDF objects indexed by a key (document-id).
    <document-id1> <tf-idf-object1>
    <document-id2> <tf-idf-object2>
    ...
    """
    try:
        key = read_key(file_handle)
        while key:
            tf_idf = TFIDF()
            try:
                tf_idf.read(file_handle)
            except RuntimeError:
                raise
            yield key, tf_idf
            key = read_key(file_handle)
    finally:
        file_handle.close()
