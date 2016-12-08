#!/usr/bin/env python

# Copyright 2016 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

# ======== Prepare data/local directory for babel data with unicode tags =======
# This script creates all files in the data/local directory for babel formats,
# except for the filtered_lexicon.txt file which is created by the
# make_lexicon_subset.sh script.
#
# This script basically takes the place of the prepare_lexicon.pl script. It
# creates the following files.
#
# 1. lexicon.txt (via local/lexicon/encodescript.py which actually happens prior
#    to running this script.
# 2. nonsilence_phones.txt
# 3. silence_phones.txt
# 4. optional_silence.txt
# 5. extra_questions.txt
# =============================================================================

from __future__ import print_function
import codecs
import sys
import os
import pdb

SKIP = ("","''","<",">","#")

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
  with codecs.open(table,"r","utf-8") as fp:

    # Get the relevant header columns for extracting the graphemes used in the lexicon
    # --------------------------------------------------------------------------------
    header = fp.readline()
    idx = [i for i,j in enumerate(header.strip().split('\t')) if j.startswith("MAP")]

    # -------------------------------------------------------------------------
    # Extract all unique graphemes. Place into formats ...
    # 1. unicode_graphemes = [g1, g2, g3, ... , gN]
    #
    # 2. Graphemes dict as keys for each base (without tags) grapheme along with
    # all distinct graphmes starting with the base grapheme.
    # phones_dict = {p1: p1_with_tags_1, p1_with_tags_2, ... , p2: ... }
    # --------------------------------------------------------------------------
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
    return unicode_graphemes,graphemes_dict

def write_nonsilence_phones(graphemes_dict,nonsilence_phones):
  with codecs.open(nonsilence_phones,"w","utf-8") as fp:
    # First write <hes>
    fp.write("<hes>\n")

    # Write each base grapheme with all tags on the same line
    for base_grapheme in sorted(graphemes_dict.keys()):
      line = ""
      for grapheme in graphemes_dict[base_grapheme]:
        line += grapheme + " "
      fp.write("%s\n" % line.strip())

def write_extra_questions(unicode_graphemes, graphemes_dict, tags, extra_questions):
  with codecs.open(extra_questions,"w","utf-8") as fp:
    # Write all unique "phones" but really graphemes in this case, plus <hes>
    # to a single line.
    fp.write("<hes> ")
    for g in unicode_graphemes:
      fp.write("%s " % g)

    # Write the non-speech words on a new line
    fp.write("\n<oov> <sss> <vns> SIL\n")

    # Write all possible phone_tag combinations that occur in the lexicon
    for tag in tags:
      for g in graphemes_dict.keys():
        tagged_grapheme = "_".join([g,tag])
        if(tagged_grapheme in graphemes_dict[g]):
          fp.write("%s " % tagged_grapheme)

      fp.write("\n")

def main():
  #  --------------- Extract unicode_graphemes from the table -----------------
  if(len(sys.argv) < 2):
    print("Usage: local/prepare_unicode_lexicon.txt <lexicon_table>")
    sys.exit(0)

  table = sys.argv[1]
  unicode_graphemes,graphemes_dict = extract_graphemes(table)

  # ---------------- Prepare the directory data/local and a few files ---------
  # Create the data/local directory if it does not yet exist
  if not os.path.exists("data/local"):
    os.makedirs("data/local")

  # Write the slience_phones.txt file
  with open("data/local/silence_phones.txt","w") as fp:
    fp.write("<oov>\n<sss>\n<vns>\nSIL\n")

  # Write the optional_silence.txt file
  with open("data/local/optional_silence.txt","w") as fp:
    fp.write("SIL\n")

  # --------------- Write the nonsilence_phones.txt file ----------------------
  write_nonsilence_phones(graphemes_dict,"data/local/nonsilence_phones.txt")

  # ------------------------- Extract tags ------------------------------
  tags = []
  for g in unicode_graphemes:
    # Only consider graphemes with tags
    g_tags = g.split("_")
    if(len(g_tags) > 1):
      tag = "_".join(g_tags[1:])
      if(tag not in tags):
        tags.append(tag)

  # --------------- Write the extra questions file ----------------------------
  write_extra_questions(unicode_graphemes, graphemes_dict, tags, "data/local/extra_questions.txt")

if __name__ == "__main__":
  main()
