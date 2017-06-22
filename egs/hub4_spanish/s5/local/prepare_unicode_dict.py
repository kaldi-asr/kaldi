#!/usr/bin/env python

# Copyright 2016 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

# ======= Prepare data/local directory for babel data with unicode tags ======
# This script creates all files in the data/local directory needed to generate
# the kaldi-lang directory. It does not generate the lexicon itself -- for that
# see the script lexicon/make_unicode_lexicon.py. It expects 
#
# ============================================================================

from __future__ import print_function
import codecs
import sys
import os
import argparse

SKIP = ("", "''", "<", ">", "#")


# Extract a sorted set of distinct unicode graphemes from the lexicon
def extract_graphemes(table):
    '''
    Extract a sorted set of distinct unicode graphemes from the lexicon.

    Usage: extract_graphemes(PATH_TO_LEXICON_TABLE)

    Arguments:
        table -- path to the lexicon table output by make_unicode_lexicon.py

    Output:
        unicode_graphemes -- the sorted set of distinct unicode graphemes
                             that occurred in the lexicon.
    '''
    with codecs.open(table, "r", "utf-8") as fp:

        # Get relevant header columns for extracting graphemes used in lexicon
        # --------------------------------------------------------------------
        header = fp.readline()
        idx = []
        for i, j in enumerate(header.strip().split('\t')):
            if j.startswith("MAP"):
                idx.append(i)

        # --------------------------------------------------------------------
        # Extract all unique graphemes. Place into formats ...
        # 1. unicode_graphemes = [g1, g2, g3, ... , gN]
        #
        # 2. Grapheme dict as keys for each base (without tags) grapheme along
        # with all distinct graphmes starting with the base grapheme.
        # phones_dict = {p1: p1_with_tags_1, p1_with_tags_2, ... , p2: ... }
        # --------------------------------------------------------------------
        unicode_graphemes = []
        graphemes_dict = {}
        for line in fp:
            for i in idx:
                grapheme = line.strip().split('\t')[i]
                if grapheme not in SKIP:
                    unicode_graphemes.append(grapheme)

    # Create the sorted set of distinct unicode graphemes in the lexicon
    unicode_graphemes = sorted(set(unicode_graphemes))
    for g in unicode_graphemes:
        base_graph = g.split("_")[0]
        if(base_graph not in graphemes_dict.keys()):
            graphemes_dict[base_graph] = []

        graphemes_dict[base_graph].append(g)

    return unicode_graphemes, graphemes_dict


def write_nonsilence_phones(graphemes_dict, nonsilence_phones,
                            extraspeech=None):
    with codecs.open(nonsilence_phones, "w", "utf-8") as fp:
        try:
            with codecs.open(extraspeech, "r", "utf-8") as f:
                for line in f:
                    line_vals = line.strip().split()
                    fp.write("%s\n" % line_vals[1])
        except (IOError, TypeError):
            pass

        # Write each base grapheme with all tags on the same line
        for base_grapheme in sorted(graphemes_dict.keys()):
            line = ""
            for grapheme in graphemes_dict[base_grapheme]:
                line += grapheme + " "
            fp.write("%s\n" % line.strip())


def write_extra_questions(unicode_graphemes, graphemes_dict, tags,
                          extra_questions, nonspeech=None, extraspeech=None):
    with codecs.open(extra_questions, "w", "utf-8") as fp:
        # Write all unique "phones" but graphemes in this case, plus <hes> to a
        # single line.

        # Write the extraspeech
        try:
            with codecs.open(extraspeech, "r", "utf-8") as f:
                for line in f:
                    line_vals = line.strip().split()
                    fp.write("%s " % line_vals[1])
        except (IOError, TypeError):
            pass

        for g in unicode_graphemes:
            fp.write("%s " % g)
        fp.write("\n")

        # Write the nonspeech
        try:
            with codecs.open(nonspeech, "r", "utf-8") as f:
                for line in f:
                    line_vals = line.strip().split()
                    fp.write("%s " % line_vals[1])
                fp.write("\n")
        except (IOError, TypeError):
            pass

        # Write all possible phone_tag combinations that occur in the lexicon
        for tag in tags:
            for g in graphemes_dict.keys():
                tagged_grapheme = "_".join([g, tag])
                if(tagged_grapheme in graphemes_dict[g]):
                    fp.write("%s " % tagged_grapheme)
            fp.write("\n")


def main():
    #  --------------- Extract unicode_graphemes from the table --------------
    if(len(sys.argv[1:]) == 0):
        print("Usage: local/prepare_unicode_dict.pu <lex_table> <lex_dir>")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("table", help="Table containing all information about"
                        " how to map unicode graphemes to unicode descriptors"
                        " See local/lexicon/make_unicode_lexicon.py for"
                        " description of the table format")
    parser.add_argument("lex_dir", help="Directory to which all files"
                        " should be written")
    parser.add_argument("--nonspeech", help="File with map of nonspeech words",
                        action="store", default=None)
    parser.add_argument("--extraspeech", help="File with map of extraspeech"
                        " words", action="store", default=None)
    args = parser.parse_args()
    unicode_graphemes, graphemes_dict = extract_graphemes(args.table)

    # ---------------- Prepare the directory data/local and a few files ------
    # Create the data/local directory if it does not yet exist
    if not os.path.exists(args.lex_dir):
        os.makedirs(args.lex_dir)

    # Write the slience_phones.txt file
    with open(os.path.join(args.lex_dir, "silence_phones.txt"), "w") as fo:
        with open(args.nonspeech, "r") as fi:
            for line in fi:
                line_vals = line.strip().split()
                fo.write("%s\n" % line_vals[1])

    # Write the optional_silence.txt file
    with open(os.path.join(args.lex_dir, "optional_silence.txt"), "w") as fp:
        fp.write("SIL\n")

    # --------------- Write the nonsilence_phones.txt file -------------------
    write_nonsilence_phones(graphemes_dict,
                            os.path.join(args.lex_dir, "nonsilence_phones.txt"),
                            extraspeech=args.extraspeech)

    # ------------------------- Extract tags ---------------------------------
    tags = []
    for g in unicode_graphemes:
        # Only consider graphemes with tags
        g_tags = g.split("_")
        if(len(g_tags) > 1):
            tag = "_".join(g_tags[1:])
            if(tag not in tags):
                tags.append(tag)

    # --------------- Write the extra questions file -------------------------
    write_extra_questions(unicode_graphemes, graphemes_dict, tags,
                          os.path.join(args.lex_dir, "extra_questions.txt"),
                          nonspeech=args.nonspeech,
                          extraspeech=args.extraspeech)


if __name__ == "__main__":
    main()
