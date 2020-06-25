#!/usr/bin/env python

# Copyright 2016 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

# ======= Prepare dictionary directory (e.g. data/local) from lexicon.txt =====
# This script takes a valid kaldi format lexicon (lexicon.txt) as input and
# from it creates the rest of the files in the dictionary directory.
# The lexicon.txt can be created from, 
# 
#    local/lexicon/make_unicode_lexicon.py <wordlist> <lexicon> <grapheme_map> 
#
# using a list of words found in the training language(s) for example. But any
# valid kaldi format lexicon should work.
#
# The files created are:
#
# 1. nonsilence_phones.txt
# 2. silence_phones.txt
# 3. optional_silence.txt
# 4. extra_questions.txt
#
# You should probably just create these files in the same directory as you
# created lexicon.txt (via local/lexicon/make_unicode_lexicon.py), otherwise
# you will have to copy lexicon.txt into the output directory of this script.
#
# Since silence and non-silence phonemes are treated separately, this script
# requires that the list of words whose pronunciations contain silence phones,
# (phones that should be in silence_phones.txt), be entered using the 
# 
#   --silence-lexicon <path-to-silence-lexicon.txt> 
#
# option. If the option is not provided, two dictionary entries are created 
# automatically: 
# 1. !SIL SIL
# 2. <unk> <oov>
#
# corresponding to entries for silence and unknown words respectively.
#
#
# Any tokens in lexicon.txt occurring in columns other than the first are
# considered to represent an acoustic unit. The set of all such tokens, that
# do not also occur in silence_lexicon.txt (or that are not SIL), are 
# written to nonsilence_phones.txt. Each line in nonsilence_phones.txt
# corresponds to an acoustic unit and its tagged versions seen in the lexicon.
# A tagged acoustic unit is represented in lexicon.txt as a token followed by an
# underscore and the name of the tag. 
#
# Example: a a_tag1 a_tag2 a_tag1_tag2
# 
# These tags determine the extra questions
# to ask in a later tree-building stage and are written to extra_questions.txt.
#
# The set of all such tokens that occur in silence_lexicon.txt are written to
# silence_phones.txt.
#
# The acoustic units used in the lexicon can be phonemes,
# graphemic-acoustic-units (units derived from a word's orthography in segmental
# writing systems), units discovered from an unsupervised clustering procedure,
# or other. For the purposes of this script, however, they are all referred to
# as phonemes.
#
# # ============================================================================

from __future__ import print_function
import codecs
import sys
import os
import argparse


# Extract a sorted set of distinct phonemes from the lexicon
def extract_phonemes(lexicon):
    '''
        Extract a sorted set of distinct phonemes from the lexicon.

        Usage: extract_phones(dictionary of lexical entries)

        Arguments:
            lexicon -- dictionary lexical entries

        Output:
            phonemes      -- the sorted set of distinct phonemes
                             that occurred in the lexicon.
            phonemes_dict -- the dictionary of keys as untagged base
                             phonemes, and values as all types of tags,
                             including untagged versions of the base phoneme.
    '''
    # Read all baseform units into dictionary with {a: [a, a_1, a_2],
    #                                               b: [b_1, b_3], ...}
    phonemes_dict = {}
    for word, pron in lexicon.items():
        for p in pron.split():
            try:
                base = p.split("_",1)[0]
                phonemes_dict[base] += [p]
            except KeyError:
                phonemes_dict[base] = [p]

    # Makes sure there are no repeats in the list
    phonemes_dict = {k: set(v) for k, v in phonemes_dict.items()}

    # Get all unique phonemes
    phonemes = []
    for v in phonemes_dict.values():
        for p in v:
            phonemes.append(p)

    phonemes = sorted(set(phonemes))

    return phonemes, phonemes_dict


def write_phonemes(phonemes_dict, phonesfile):
    with codecs.open(phonesfile, "w", "utf-8") as fp:
        # Write each base phoneme with all tags on the same line
        for base_phoneme in sorted(phonemes_dict.keys()):
            line = ""
            for phoneme in sorted(phonemes_dict[base_phoneme]):
                line += phoneme + " "
            fp.write("%s\n" % line.strip())


def write_extra_questions(nonsil_phonemes, nonsil_phonemes_dict,
                          sil_phonemes, sil_phonemes_dict,
                          tags, extra_questions):
    with codecs.open(extra_questions, "w", "utf-8") as fp:
        # Write all unique "nonsilence_phones" to a single line.
        for p in nonsil_phonemes:
            fp.write("%s " % p)
        fp.write("\n")

        # Write the silence_lexicon
        for p in sil_phonemes:
            fp.write("%s " % p)
        fp.write("\n")

        # Write all possible phone_tag combinations that occur in the lexicon
        for tag in tags:
            for p in nonsil_phonemes_dict.keys():
                tagged_phoneme = "_".join([p, tag])
                if(tagged_phoneme in nonsil_phonemes_dict[p]):
                    fp.write("%s " % tagged_phoneme)
            for p in sil_phonemes_dict.keys():
                tagged_phoneme = "_".join([p, tag])
                if(tagged_phoneme in sil_phonemes_dict[p]):
                    fp.write("%s " % tagged_phoneme)
            fp.write("\n")


def main():
    # ----------------- Parse input arguments ---------------------------
    if(len(sys.argv[1:]) == 0):
        print("Usage: local/prepare_unicode_lexicon.txt <lexicon>"
              " <lexicon_dir>", file=sys.sterr)
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("lexicon", help="A kaldi format lexicon.")
    parser.add_argument("lexicon_dir", help="Directory to which all files"
                        " should be written")
    parser.add_argument("--silence-lexicon", help="File with silence words "
                        "and tab-separated pronunciations", action="store",
                        default=None)
    args = parser.parse_args() 

    # ---------------- Prepare the dictionary directory -----------------
    # Create the data/local(/dict) directory for instance if it does not exist
    if not os.path.exists(args.lexicon_dir):
        os.makedirs(args.lexicon_dir)

    # ----------- Extract silence words and phonemes -----------------
    sil_lexicon = {}
    try:
        with codecs.open(args.silence_lexicon, "r", encoding="utf-8") as fi:
            for line in fi:
                sil_word, sil_pron = line.strip().split(None, 1)
                sil_lexicon[sil_word] = sil_pron
    except TypeError:
        # Default silence token and pron (required for using optional silence)
        # Also default unk token and pron.
        sil_lexicon = {'!SIL': 'SIL', '<unk>': '<oov>'}
    except IOError:
        print("Could not find file", args.silence_lexicon)
        sys.exit(1)

    sil_phonemes, sil_phonemes_dict = extract_phonemes(sil_lexicon)

    # This catches the optional silence symbol, which we want to include
    if 'SIL' not in sil_phonemes:
        sil_phonemes = sil_phonemes.union(['SIL'])
        sil_phonemes_dict['SIL'] = ['SIL']

    # ---------- Extract nonsilence words and phonemes ---------------
    nonsil_lexicon = {}
    try:
        with codecs.open(args.lexicon, "r", encoding="utf-8") as fi:
            for line in fi:
                word, pron = line.strip().split(None, 1)
                if word not in sil_lexicon:
                    nonsil_lexicon[word] = pron
    except TypeError:
        print("Invalid lexicon argument")
        sys.exit(1)
    except IOError:
        print("Could not find file", args.lexicon)

    nonsil_phonemes, nonsil_phonemes_dict = extract_phonemes(nonsil_lexicon)
    
    # Write silence_phones.txt
    write_phonemes(sil_phonemes_dict,
                   os.path.join(args.lexicon_dir, "silence_phones.txt"))

    # Write nonsilence_phones.txt
    write_phonemes(nonsil_phonemes_dict,
                   os.path.join(args.lexicon_dir, "nonsilence_phones.txt"))

    # Write the optional_silence.txt file
    with open(os.path.join(args.lexicon_dir, "optional_silence.txt"), "w") as fp:
        fp.write("SIL\n")

    # ------------------------- Extract tags ---------------------------------
    tags = []
    for p in set(nonsil_phonemes).union(set(sil_phonemes)):
        # Only consider phonemes with tags
        p_tags = p.split("_")
        if(len(p_tags) > 1):
            tag = "_".join(p_tags[1:])
            if(tag not in tags):
                tags.append(tag)

    # --------------- Write the extra questions file -------------------------
    write_extra_questions(nonsil_phonemes, nonsil_phonemes_dict,
                          sil_phonemes, sil_phonemes_dict, tags,
                          os.path.join(args.lexicon_dir, "extra_questions.txt"))


if __name__ == "__main__":
    main()
