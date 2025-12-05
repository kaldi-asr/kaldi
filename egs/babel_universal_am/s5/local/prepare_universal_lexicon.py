#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

###############################################################################
#
# This script takes a kaldi formatted lexicon prepared by
#
#  local/prepare_lexicon.pl (i.e. a lexicon that uses the X-SAMPA phoneset)
#
# and makes language specific modifications to further standardize the
# lexicons across languages. These modifications are based on language speficic
# diphthong and tone files that contain a mapping from diphthongs to other
# X-SAMPA phonemes, and from the Tone markers to a standardized tone marking
# (see universal_phone_maps/tones/README.txt for more info about tone).
#
# This script returns the resulting standardized lexicon.
#
###############################################################################

from __future__ import print_function
import argparse
import codecs
import os


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("olexicon", help="Output kaldi format lexicon.")
    parser.add_argument("lexicon", help="Kaldi format lexicon")
    parser.add_argument("diphthongs", help="File with diphthong mapping")
    parser.add_argument("tones", help="File with tone mapping")
    return parser.parse_args()


def main():
    args = parse_input()

    # load diphthong map
    dp_map = {}
    try:
        with codecs.open(args.diphthongs, "r", encoding="utf-8") as f:
            for l in f:
                phone, split_phones = l.split(None, 1)
                dp_map[phone] = split_phones.split()
    except IOError:
        dp_map = {}

    # load tone map
    tone_map = {}
    try:
        with codecs.open(args.tones, "r", encoding="utf-8") as f:
            for l in f:
                tone, split_tones = l.split(None, 1)
                tone_map[tone] = split_tones.split()
    except IOError:
        tone_map = {}

    # Process lexicon
    lexicon_out = []
    with codecs.open(args.lexicon, "r", encoding="utf-8") as f:
        # Read in each line storing the word, and pronunciation. Split each
        # pronunciation into its consituent phonemes. For each of these
        # phonemes, recover any tags ("x_TAG"), and replace them if needed
        # with a new tag symbol(s) if required by the tone mapping.
        for l in f:
            word, pron = l.split(None, 1)
            new_pron = ""
            for p in pron.split():
                try:
                    p, tags = p.split("_", 1)
                    # Process tags
                    tags = tags.split("_")
                    new_tags = []
                    for t in tags:
                        try:
                            new_tags += tone_map[t]
                        except KeyError:
                            new_tags += t
                except ValueError:
                    new_tags = []

                # Process diphthongs
                try:
                    new_phones = dp_map[p]
                except KeyError:
                    new_phones = [p]

                # Join tags and phones
                for nph in new_phones:
                    new_pron += "_".join([nph] + new_tags) + " "

            lexicon_out.append((word, new_pron))

    # Write new lexicon. Check output path and create any necessary
    # intermediate directories
    if (not os.path.exists(os.path.dirname(args.olexicon))):
        os.makedirs(os.path.dirname(args.olexicon))

    with codecs.open(args.olexicon, "w", encoding="utf-8") as f:
        for w, new_pron in lexicon_out:
          print(u"{}\t{}".format(w, new_pron.strip()), file=f)


if __name__ == "__main__":
    main()
