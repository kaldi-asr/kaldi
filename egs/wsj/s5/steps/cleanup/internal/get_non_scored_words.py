#!/usr/bin/env python3

# Copyright 2016   Vimal Manohar
#           2016   Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

from __future__ import print_function
import argparse
import logging
import operator
import os
import sys
from collections import defaultdict

import io
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# If you supply the <lang> directory (the one that corresponds to
# how you decoded the data) to this script, it assumes that the <lang>
# directory contains phones/align_lexicon.int, and it uses this to work
# out a reasonable guess of the non-scored phones, based on which have
# a single-word pronunciation that maps to a silence phone.
# It then uses the words.txt to work out the written form of those words.

parser = argparse.ArgumentParser(
    description = "This program works out a reasonable guess at a list of "
    "non-scored words (words that won't affect the WER evaluation): "
    "things like [COUGH], [NOISE] and so on.  This is useful because a list of "
    "such words is required by some other scripts (e.g. modify_ctm_edits.py), "
    "and it's inconvenient to have to specify the list manually for each language. "
    "This program writes out the words in text form, one per line.")

parser.add_argument("lang", type = str,
                    help = "The lang/ directory.  This program expects "
                    "lang/words.txt and lang/phones/silence.int and "
                    "lang/phones/align_lexicon.int to exist, and will use them to work "
                    "out a reasonable guess of the non-scored words  (as those whose "
                    "pronunciations are a single phone in the 'silphones' list)")

args = parser.parse_args()

non_scored_words = set()


def read_lang(lang_dir):
    global non_scored_words

    if not os.path.isdir(lang_dir):
        logger.error("expected lang/ directory %s to "
                     "exist.", lang_dir)
        raise RuntimeError

    for f in [ '/words.txt', '/phones/silence.int', '/phones/align_lexicon.int' ]:
        if not os.path.exists(lang_dir + f):
            logger.error("expected file %s%s to exist.", lang_dir, f)
            raise RuntimeError

    # read silence-phones.
    try:
        silence_phones = set()
        for line in open(lang_dir + '/phones/silence.int').readlines():
            silence_phones.add(int(line))
    except Exception:
        logger.error("problem reading file "
                     "%s/phones/silence.int", lang_dir)
        raise

    # read align_lexicon.int.
    # format is: <word-index> <word-index> <phone-index1> <phone-index2> ..
    # We're looking for line of the form:
    # w w p
    # where w > 0 and p is in the set 'silence_phones'
    try:
        silence_word_ints = set()
        for line in open(lang_dir + '/phones/align_lexicon.int').readlines():
            a = line.split()
            if len(a) == 3 and a[0] == a[1] and int(a[0]) > 0 and \
                    int(a[2]) in silence_phones:
                silence_word_ints.add(int(a[0]))
    except Exception:
        logger.error("problem reading file %s/phones/align_lexicon.int",
                     lang_dir)
        raise

    try:
        for line in open(lang_dir + '/words.txt', encoding='utf-8').readlines():
            [ word, integer ] = line.split()
            if int(integer) in silence_word_ints:
                non_scored_words.add(word)
    except Exception:
        logger.error("problem reading file %s/words.txt.int", lang_dir)
        raise

    if not len(non_scored_words) == len(silence_word_ints):
        raise RuntimeError("error getting silence words, len({0}) != len({1})"
                           "".format(non_scored_words, silence_word_ints))
    for word in non_scored_words:
        print(word)


read_lang(args.lang)
